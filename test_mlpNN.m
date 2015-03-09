function [cc] = test_mlpNN(model, input, target)
    
    % racunamo izlaz iz mreze i usporedjujemo ga sa zadanim izlazom

    [ntest, nOutLayer] = size(target);
    
    % izlaz neuronske mreze
    output = zeros(ntest, nOutLayer);
    
    for i = 1:ntest
        temp = input(i,:); % propagacija funkcijskog signala unaprijed kroz mrezu
        for j = 1:length(model.weights)
            temp = temp * model.weights{j} + model.biases{j}; % izracunaj izlaz
            temp = 1./(1+exp(-temp)); % aktivacijska funkcija neurona je sigmoidalna
        end
            % zapamti zadnji izlaz
            M = max(temp); % odredi najveci iznos izlaza i klasificiraj uzorak
            for k = 1:nOutLayer
                if(temp(:,k) == M)   
                    output(i,k) = 1;
                end
            end
    end
    
    sum = 0;
    
    for i = 1:ntest
        if(output(i,:) == target(i,:))
            sum = sum + 1;
        end
    end
    
    cc = sum./ntest;  
end
