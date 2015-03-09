function [model, cc] = train_mlpNN(input, target, hidden, iterations, learning_rate, momentum)

    % inicijaliziraj model
    model = [];
    model.learning_rate = learning_rate;
    model.momentum = momentum; % ucenje se moze ubrzati koristenjem momenta (izbjegavanje lokalnih minimuma)
    
    % karakteristike ulaza i izlaza
    [ntrain, nInLayer] = size(input);
    [~, nOutLayer] = size(target);
    
    % zapamti koliko je neurona u svakom sloju
    nNeurons = [nInLayer hidden nOutLayer];
    nNeurons(nNeurons == 0) = []; % makni sloj s 0 neurona
     
    % skupova težina je ukupno (broj slojeva - 1)
    nTransitions = length(nNeurons)-1;
    
    % postavi tezine izmedju slojeva na slucajno odabrane vrijednosti
    for i = 1:nTransitions
        model.weights{i} = randn(nNeurons(i),nNeurons(i+1));
        % matrica tezina ima R redaka, gdje je R broj ulaznih neurona u sloj,
        % te C stupaca, gdje je C broj izlaznih neurona
        model.biases{i} = randn(1,nNeurons(i+1)); % prag
        model.lastdelta{i} = 0; % za korekciju tezina
    end
    
    for i = 1:iterations % ponovi postupak za trening skup (iterations) puta
        order = randperm(ntrain);  % redoslijed predstavljanja uzoraka za ucenje
        for j = 1:ntrain
            % ovdje ucimo mrezu i azuriramo tezine
            model = update_mlpNN(model, input(order(j),:), target(order(j),:));
        end
    end
    
    % provjeri uspjesnost na trening skupu nakon ucenja
    cc = test_mlpNN(model, input, target);
end
        