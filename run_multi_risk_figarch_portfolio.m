function [portfolio_returns_all, weights_all, risk_metrics] = run_multi_risk_figarch_portfolio(returns, window_size, fallback_window, max_iter, optimization_type, custom_options)
%RUN_MULTI_RISK_FIGARCH_PORTFOLIO Optimizes portfolio weights using FIGARCH volatility models and multiple VaR risk measures
%
% SYNTAX:
%   [portfolio_returns_all, weights_all, risk_metrics] = run_multi_risk_figarch_portfolio(returns, window_size, fallback_window, max_iter, optimization_type, custom_options)
%
% INPUTS:
%   returns - Table or matrix of returns with assets in columns and time in rows
%   window_size - Rolling window size for model estimation (default: 250)
%   fallback_window - Window size for fallback volatility estimation (default: 30)
%   max_iter - Maximum number of iterations for FIGARCH estimation (default: 6000)
%   optimization_type - Risk measure for portfolio optimization:
%      'tstudent' - VaR with Student's t-distribution
%      'cornish-fisher' - VaR using Cornish-Fisher expansion
%      'historical-var' - Historical VaR computed directly from past data
%      'all' - Runs all VaR methods and returns multi-column results
%   custom_options - Structure with customized parameters (optional)
%      .alpha - Confidence level for VaR (default: 0.99)
%      .nu - Degrees of freedom for t-distribution (default: estimated from data)
%      .figarchOrder - Order parameters for the FIGARCH model [p,q,d] (default: [1,1,0.4])
%      .lambda - Trade-off parameter between risk and return (default: 0.1)
%      .cov_method - Covariance estimation method ('diagonal', 'constant') (default: 'diagonal')
%      .rebalancing_frequency - Portfolio rebalancing frequency (default: 1)
%
% OUTPUTS:
%   portfolio_returns_all - Table of portfolio returns for each optimization method
%   weights_all - Table of portfolio weights for each asset over time
%   risk_metrics - Structure containing risk and performance metrics for each method
%
% EXAMPLE FLOWCHART:
%   
%   % START
%   │
%   ├──> Set default values if inputs are missing
%   │
%   ├──> Prepare data:
%   │     • Convert to table if necessary
%   │     • Extract asset names and dates
%   │
%   ├──> Choose risk optimization methods (single or multiple)
%   │
%   ├──> Prepare rolling window data cache
%   │
%   ├──> If needed:
%   │     • Estimate t-Student degrees of freedom (nu)
%   │
%   ├──> Set up parallel computing pool
%   │
%   ├──> LOOP OVER ALL TIME STEPS:
%   │        ↓
%   │   If no rebalancing:
%   │        • Use previous weights to avoid recomputation
%   │   Else:
%   │       ├──> Estimate asset volatility via FIGARCH (or fallback to historical)
%   │       ├──> Build covariance matrix
%   │       ├──> LOOP OVER RISK METHODS:
%   │               ↓
%   │          • Define objective function
%   │          • Optimize portfolio weights with fmincon
%   │          • Save portfolio returns and weights
%   │
%   ├──> End of loop:
%   │     • Interpolate missing data if necessary
%   │
%   ├──> Build final output tables (returns, weights)
%   │
%   ├──> Compute performance metrics
%   │
%   └──> Print performance summary
%         ↓
%       END





    % Input validation and default parameter setting
    if nargin < 2 || isempty(window_size), window_size = 250; end
    if nargin < 3 || isempty(fallback_window), fallback_window = 30; end
    if nargin < 4 || isempty(max_iter), max_iter = 6000; end
    if nargin < 5 || isempty(optimization_type), optimization_type = 'tstudent'; end
    if nargin < 6, custom_options = struct(); end
    
    % Default options
options = struct();
options.alpha = 0.99;                         % Confidence level for VaR
options.nu = [];                              % Degrees of freedom (estimated if empty)
options.figarchOrder = [1, 1, 0.4];            % FIGARCH model orders [p,q,d]
options.lambda = 0.1;                          % Risk-return trade-off parameter
options.cov_method = 'diagonal';                % Covariance estimation method
options.rebalancing_frequency = 1;              % Rebalancing every N periods
options.min_sample_size = 50;                   % Minimum sample size for reliable estimation
options.differentiate_methods = true;           % Force methods to use different approaches
options.J = 100;                                % Truncation parameter for FIGARCH

    
    % Overwrite custom options
    fn = fieldnames(custom_options);
    for i = 1:length(fn)
        options.(fn{i}) = custom_options.(fn{i});
    end
    
    
    fprintf('Iniziando ottimizzazione del portafoglio usando modelli FIGARCH e misure di VaR\n');
    fprintf('Metodo di ottimizzazione: %s\n', optimization_type);
    fprintf('Dimensione finestra: %d, Finestra fallback: %d\n', window_size, fallback_window);
    fprintf('Livello di confidenza (alpha): %.4f\n', options.alpha);
    
    
    if ~istable(returns)
        warning("Input non è una tabella, lo converto automaticamente.");
        returns = array2table(returns, 'VariableNames', strcat("Asset", string(1:size(returns,2))));
    end
    
    
    dates = returns.Properties.RowNames;
    if isempty(dates)
        dates = string((1:height(returns))');
    else
        dates = string(dates);
    end
    
    
    numeric_vars = varfun(@isnumeric, returns, 'OutputFormat', 'uniform');
    strategies = returns.Properties.VariableNames(numeric_vars);
    returns_data = returns(:, strategies);
    [T, N] = size(returns_data);
    run_all = strcmp(optimization_type, 'all');
    if run_all
        methods = {'tstudent', 'cornish-fisher', 'historical-var'};
        num_methods = length(methods);
    else
        methods = {optimization_type};
        num_methods = 1;
    end
       portfolio_returns_matrix = nan(T - window_size, num_methods);
       weights_tensor = nan(T - window_size, N, num_methods);
    
    
    if any(strcmp(methods, 'tstudent')) && isempty(options.nu)
    % Setup for Student's t-distribution if needed
        try
            options.nu = estimate_t_distribution_dof(table2array(returns_data));
            fprintf('Gradi di libertà stimati per distribuzione t: %.2f\n', options.nu);
        catch
            options.nu = 6;  % Default fallback
            fprintf('Utilizzo gradi di libertà di default: %.2f\n', options.nu);
        end
    end
    
    % Comput. quantiles
    t_quantile = tinv(options.alpha, options.nu);
    
    
    %parallel pool
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool('local', min(feature('numcores'), 8));
    end
    
    
    window_data_cache = cell(T-window_size+1, 1);
    for t = window_size:T
        window_data_cache{t-window_size+1} = table2array(returns_data(t-window_size+1:t, :));
    end
    
    % Begin optim.loop
    fprintf('Avanzamento ottimizzazione:     ');
    for t_idx = 1:(T-window_size)
        %Print the status
        if mod(t_idx, max(1, floor((T-window_size)/20))) == 0
            fprintf('\b\b\b\b%3d%%', round(100*t_idx/(T-window_size)));
        end
        
        t = window_size + t_idx - 1;
        window_data = window_data_cache{t_idx};
        
        
        if mod(t_idx-1, options.rebalancing_frequency) ~= 0 && t_idx > 1
            for m = 1:num_methods
                weights_tensor(t_idx, :, m) = weights_tensor(t_idx-1, :, m);
                if t+1 <= T
                    next_returns = table2array(returns_data(t+1, :))';
                    portfolio_returns_matrix(t_idx, m) = weights_tensor(t_idx, :, m) * next_returns;
                end
            end
            continue;
        end
        vol_estimates = nan(N, 1);
        mu_estimates = nan(N, 1);
        skew_estimates = nan(N, 1);
        kurt_estimates = nan(N, 1);
        figarch_diagnostics = cell(N, 1);
        
        parfor i = 1:N
            serie = window_data(:, i);
            serie = serie(~isnan(serie));
            
            try
                
                if length(serie) < options.min_sample_size
                    error('Serie troppo corta per stima FIGARCH');
                end
                
                % Stima modello FIGARCH
                [parameters, sigma2_hat, residuals, info] = stima_figarch_semplificato(...
                    serie, options.J, max_iter, options.figarchOrder);
                
                if info.exitflag > 0
                    vol_estimates(i) = sqrt(sigma2_hat(end));
                    mu_estimates(i) = mean(serie);
                    
                    % Per Cornish-Fisher abbiamo bisogno di momenti più alti
                    if any(strcmp(methods, 'cornish-fisher'))
                        skew_estimates(i) = skewness(residuals);
                        kurt_estimates(i) = kurtosis(residuals) - 3; % Eccesso di curtosi
                    end
                    
                    % Archivia diagnostiche
                    figarch_diagnostics{i} = struct(...
                        'parameters', parameters, ...
                        'convergence', info.exitflag, ...
                        'iterations', info.iterations, ...
                        'final_vol', sqrt(sigma2_hat(end)), ...
                        'log_likelihood', info.fval);
                else
                    error('Stima FIGARCH non converge');
                end
            catch err
                % Fallback a stime storiche
                fallback_slice = serie(max(1, end-fallback_window+1):end);
                vol_estimates(i) = std(fallback_slice, 'omitnan');
                mu_estimates(i) = mean(fallback_slice, 'omitnan');
                
                if any(strcmp(methods, 'cornish-fisher'))
                    skew_estimates(i) = skewness(fallback_slice);
                    kurt_estimates(i) = kurtosis(fallback_slice) - 3;
                end
                
                figarch_diagnostics{i} = struct('error', err.message, 'fallback', true);
            end
        end
        valid_idx = ~isnan(vol_estimates);
        if sum(valid_idx) < 2
            warning('Non abbastanza asset validi al tempo %d, salto', t);
            continue;
        end
        
        
        sigma_vec = vol_estimates(valid_idx);
        mu_vec = mu_estimates(valid_idx);
        skew_vec = skew_estimates(valid_idx);
        kurt_vec = kurt_estimates(valid_idx);
        
       
        n_valid = sum(valid_idx);
        
        % Costruisci matrice di covarianza in base al metodo selezionato
        switch options.cov_method
            case 'diagonal'
                % Usa solo diagonale (volatilità individuali)
                Sigma = diag(sigma_vec.^2);
            case 'constant'
                % Stima matrice di correlazione costante
                valid_data = window_data(:, valid_idx);
                corr_mat = corrcoef(valid_data);
                D = diag(sigma_vec);
                Sigma = D * corr_mat * D;
                
            otherwise
                % Default a diagonale
                Sigma = diag(sigma_vec.^2);
        end
        
        % Assicura definitezza positiva della matrice di covarianza
        [V, D] = eig(Sigma);
        D = diag(max(diag(D), 1e-6));
        Sigma = V * D * V';
        
        % Ottieni rendimenti del giorno successivo per calcolo performance
        if t+1 <= T
            next_returns = table2array(returns_data(t+1, valid_idx))';
        else
            next_returns = [];
        end
        
        % Salta se i rendimenti successivi contengono valori NaN
        if ~isempty(next_returns) && any(isnan(next_returns))
            continue;
        end
        
        % Ottimizza portafoglio per ogni metodo
        for m = 1:num_methods
            current_method = methods{m};
            
            % Definisci funzione obiettivo basata sul metodo
            switch current_method
                case 'tstudent'
                    % VaR con distribuzione t-Student
                    risk_func = @(w) -w'*mu_vec + t_quantile * sqrt(w' * Sigma * w * (options.nu - 2) / options.nu);    
                case 'cornish-fisher'
                    % VaR con espansione di Cornish-Fisher per non-normalità
                    % Usa implementazione avanzata che tiene conto di asimmetria e curtosi di portafoglio
                    risk_func = @(w) improved_cornish_fisher_var(w, mu_vec, Sigma, skew_vec, kurt_vec, options.alpha);
                    
                case 'historical-var'
                    % VaR storico calcolato direttamente dai dati
                    risk_func = @(w) calculate_historical_var(w, window_data(:, valid_idx), options.alpha) - options.lambda * w' * mu_vec;
                    
                otherwise
                    error('Metodo di ottimizzazione non supportato: %s', current_method);
            end
            
            % Imposta vincoli di ottimizzazione
            w0 = ones(n_valid, 1) / n_valid;
            Aeq = ones(1, n_valid);
            beq = 1;
            lb = zeros(n_valid, 1);
            
            % Configura ottimizzatore
            opt_settings = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
                'MaxIterations', 1000, 'TolFun', 1e-8, 'TolX', 1e-8);
            
            % Esegui ottimizzazione
            try
                w_opt = fmincon(risk_func, w0, [], [], Aeq, beq, lb, [], [], opt_settings);
            catch
                % Fallback a pesi uniformi
                w_opt = w0;
                warning('Ottimizzazione fallita per metodo %s al tempo %d, uso pesi uniformi', current_method, t);
            end
            
            % Calcola rendimento di portafoglio
            if ~isempty(next_returns)
                portfolio_returns_matrix(t_idx, m) = w_opt' * next_returns;
            end
            
            % Converti pesi ottimali a vettore completo di asset
            full_weights = zeros(1, N);
            full_weights(valid_idx) = w_opt';
            weights_tensor(t_idx, :, m) = full_weights;
        end
    end
    fprintf('\b\b\b\b100%%\n');
    
    % Interpola valori mancanti
    for m = 1:num_methods
        portfolio_returns_matrix(:, m) = interpola_valori_mancanti(portfolio_returns_matrix(:, m));
        for i = 1:N
            weights_tensor(:, i, m) = interpola_valori_mancanti(weights_tensor(:, i, m));
        end
    end
    
    % Crea tabelle di output
    final_dates = dates((window_size+1):T);
    
    % Crea tabella rendimenti di portafoglio
    if run_all
        portfolio_returns_all = array2table(portfolio_returns_matrix, 'VariableNames', methods);
    else
        portfolio_returns_all = array2table(portfolio_returns_matrix, 'VariableNames', {optimization_type});
    end
    portfolio_returns_all.Properties.RowNames = final_dates;
    
    % Crea tabelle pesi (una per ogni metodo)
    weights_all = cell(num_methods, 1);
    for m = 1:num_methods
        weights_all{m} = array2table(weights_tensor(:, :, m), 'VariableNames', strategies);
        weights_all{m}.Properties.RowNames = final_dates;
    end
    
    % Se solo un metodo, ritorna pesi direttamente
    if num_methods == 1
        weights_all = weights_all{1};
    end
    
    % Calcola metriche di performance
    risk_metrics = calculate_performance_metrics(portfolio_returns_matrix, methods, options.alpha);
    
    
    fprintf('\nRiepilogo Performance:\n');
    fprintf('%-15s  %10s  %10s  %10s  %10s  %10s\n', 'Metodo', 'Rend.Ann.', 'Vol.Ann.', 'Sharpe', 'MaxDD', 'CVaR 99%');
    fprintf('----------------------------------------------------------------------\n');
    
    for m = 1:num_methods
        fprintf('%-15s  %10.2f%%  %10.2f%%  %10.2f  %10.2f%%  %10.2f%%\n', ...
            methods{m}, ...
            risk_metrics.AnnualReturn(m) * 100, ...
            risk_metrics.AnnualVol(m) * 100, ...
            risk_metrics.Sharpe(m), ...
            risk_metrics.MaxDrawdown(m) * 100, ...
            risk_metrics.CVaR99(m) * 100);
    end
 
end
%% Funzioni di supporto

function [dof] = estimate_t_distribution_dof(returns)
    % Stima gradi di libertà per distribuzione t dai dati di rendimento
    
    % Standardizza rendimenti
    standardized_returns = returns ./ std(returns, 0, 1, 'omitnan');
    % Combina tutte le serie per una stima migliore
    all_returns = standardized_returns(:);
    all_returns = all_returns(~isnan(all_returns));
    
    % Definisci funzione di log-verosimiglianza negativa per distribuzione t
    t_nll = @(nu) -sum(log(tpdf(all_returns, nu)));
    
    % Ottimizza per trovare i migliori gradi di libertà (tra 2.1 e 30)
    options = optimset('Display', 'off');
    [dof, ~, exitflag] = fminbnd(t_nll, 2.1, 30, options);
    
    if exitflag <= 0
        % Fallback a default se ottimizzazione fallisce
        dof = 6;
    end
end

function [var_cf] = improved_cornish_fisher_var(w, mu, Sigma, skew, kurt, alpha)
    % Calcola VaR usando espansione Cornish-Fisher migliorata che tiene conto correttamente dei momenti di portafoglio
    
    % Media e varianza di portafoglio
    port_mu = w' * mu;
    port_vol = sqrt(w' * Sigma * w);
    
    % Calcola asimmetria standardizzata del portafoglio
    % Questa è un'approssimazione basata su momenti individuali degli asset
    port_skew = 0;
    port_kurt = 0;
    
    % Calcola asimmetria di portafoglio (approccio semplificato)
    n = length(w);
    for i = 1:n
        % Contributo all'asimmetria
        port_skew = port_skew + w(i)^3 * skew(i) * (sqrt(Sigma(i,i))^3) / (port_vol^3);
        
        % Contributo alla curtosi
        port_kurt = port_kurt + w(i)^4 * kurt(i) * (Sigma(i,i)^2) / (port_vol^4);
        
        % Aggiunge termini incrociati per curtosi (semplificato)
        for j = 1:n
            if i ~= j
                port_kurt = port_kurt + w(i)^2 * w(j)^2 * (Sigma(i,j)^2) / (port_vol^4);
            end
        end
    end
    
    % Quantile normale standard
    z_alpha = norminv(alpha);
    
    % Espansione Cornish-Fisher migliorata (fino al 4° momento)
    h1 = (z_alpha^2 - 1) / 6;
    h2 = (z_alpha^3 - 3*z_alpha) / 24;
    h3 = -(2*z_alpha^3 - 5*z_alpha) / 36;
    
    cf_quantile = z_alpha + h1*port_skew + h2*port_kurt + h3*port_skew^2;
    
    % Calcola VaR
    var_cf = -port_mu + port_vol * cf_quantile;
end

function var_hist = calculate_historical_var(w, historical_returns, alpha)
    % Calcola il VaR storico per un dato portafoglio
    
    % Controllo input
    if isempty(historical_returns) || size(historical_returns, 1) < 10
        var_hist = Inf;
        return;
    end
    
    % Calcola rendimenti storici del portafoglio
    portfolio_returns = historical_returns * w;
    
    % Ordina rendimenti e trova quantile corrispondente
    sorted_returns = sort(portfolio_returns);
    index = max(1, ceil(size(historical_returns, 1) * (1-alpha)));
    
    % VaR è il negativo del rendimento al quantile specificato
    var_hist = -sorted_returns(index);
end

function [h, resid] = simulate_garch11(returns, beta, alpha, omega)
    % Simulazione semplice GARCH(1,1)
    T = length(returns);
    h = zeros(T, 1);
    resid = zeros(T, 1);
    
    % Inizializza con varianza campionaria
    h(1) = var(returns(1:min(20,T)));
    resid(1) = returns(1);
    
    % Ricorsione GARCH
    for t = 2:T
        h(t) = omega + alpha * resid(t-1)^2 + beta * h(t-1);
        resid(t) = returns(t);
    end
end

function metrics = calculate_performance_metrics(returns, methods, alpha)
    % Calcola metriche di performance per ogni portafoglio
    
    num_methods = length(methods);
    metrics = struct();
    
    % Giorni di trading per anno (approssimativo)
    trading_days = 252;
    
    % Inizializza metriche
    metrics.AnnualReturn = zeros(num_methods, 1);
    metrics.AnnualVol = zeros(num_methods, 1);
    metrics.Sharpe = zeros(num_methods, 1);
    metrics.MaxDrawdown = zeros(num_methods, 1);
    metrics.CVaR99 = zeros(num_methods, 1);
    metrics.Return_CVaR_Ratio = zeros(num_methods, 1);
    metrics.VaR99 = zeros(num_methods, 1); % Aggiunto VaR
    metrics.ExpectedShortfall = zeros(num_methods, 1); % Aggiunto Expected Shortfall
    
    for m = 1:num_methods
        rets = returns(:, m);
        rets = rets(~isnan(rets));
        
        if ~isempty(rets)
            % Rendimento annuo (geometrico)
            metrics.AnnualReturn(m) = prod(1 + rets)^(trading_days/length(rets)) - 1;
            
            % Volatilità annua
            metrics.AnnualVol(m) = std(rets) * sqrt(trading_days);
            
            % Sharpe ratio (assumendo tasso risk-free zero per semplicità)
            metrics.Sharpe(m) = metrics.AnnualReturn(m) / metrics.AnnualVol(m);
            
            % Massimo drawdown
            cumulative = cumprod(1 + rets);
            running_max = zeros(size(cumulative));
            running_max(1) = cumulative(1);
            for i = 2:length(cumulative)
                running_max(i) = max(running_max(i-1), cumulative(i));
            end
            drawdowns = (cumulative ./ running_max) - 1;
            metrics.MaxDrawdown(m) = min(drawdowns);
            
            % Value at Risk (VaR) al 99%
            sorted_returns = sort(rets);
            cutoff_index = ceil((1-alpha) * length(sorted_returns));
            metrics.VaR99(m) = -sorted_returns(cutoff_index);
            
            % Conditional Value at Risk (CVaR/Expected Shortfall) al 99%
            metrics.CVaR99(m) = mean(sorted_returns(1:cutoff_index));
            metrics.ExpectedShortfall(m) = metrics.CVaR99(m);
            
            % Rapporto rendimento/CVaR
            metrics.Return_CVaR_Ratio(m) = metrics.AnnualReturn(m) / abs(metrics.CVaR99(m));
        end
    end
end

function interpolated = interpola_valori_mancanti(data)
    % Interpolazione migliorata dei valori mancanti
    
    % Se tutti i valori sono mancanti, ritorna i dati originali
    if all(isnan(data))
        interpolated = data;
        return;
    end
    
    % Usa l'interp1 di MATLAB per una migliore interpolazione
    idx = 1:length(data);
    valid_idx = find(~isnan(data));
    
    if length(valid_idx) < 2
        % Non abbastanza punti per interpolazione, usa fillmissing
        interpolated = fillmissing(data, 'nearest');
    else
        % Interpolazione lineare
        interpolated = interp1(valid_idx, data(valid_idx), idx, 'linear', 'extrap');
    end
end




