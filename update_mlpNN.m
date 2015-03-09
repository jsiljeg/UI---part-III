function [model] = update_mlpNN(model, input, target)

    % funkcija se poziva jednom za svaki uzorak iz skupa za ucenje,
    % pritom se svaki put azuriraju tezine

    % aktivacijski signal neurona u svakom sloju
    activations = cell(length(model.weights)+1,1);
    activations{1} = input;
    
    % izracunaj aktivacijske signale svih slojeva neurona
    for i = 1:length(model.weights)
        % activations{i} je vektor - redak
        % model.weights{i} je matrica tezina
        % rezultat produkta je vektor - redak duljine (broj neurona u sljedecem sloju)
        temp = activations{i} * model.weights{i} + model.biases{i}; 
        activations{i+1} = 1./(1+exp(-(temp))); % aktivacijska funkcija je sigmoidalna
    end
    
    % signal greske
    errors = cell(length(model.weights),1);
    
    % propagacija signala greske unatrag kroz mrezu
    run_error = (target - activations{end}); % lokalni gradijent
    for i = length(model.weights):-1:1
        errors{i} = activations{i+1} .* (1-activations{i+1}) .* (run_error);
        run_error = errors{i} * model.weights{i}';
    end
    
    % azuriraj tezine i pragove
    for i = 1:length(model.weights)
        % azuriranje tezina temelji se na delta pravilu
        model.weights{i} = model.weights{i} + model.learning_rate * activations{i}' * errors{i};
        model.biases{i} = model.biases{i} + model.learning_rate * errors{i};
    end
end
