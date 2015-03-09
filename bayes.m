load('mnist.mat');
X=X';
trainingSize = 60000; %postavimo velicinu trening skupa

X_bin = logical(X(:, 1:trainingSize)); %pretvorimo u logical jer razlikujemo samo crne i bijele piksele
Y_bin = logical(Y);

labelsnew = labels(1:trainingSize);

%Trening

for i=1:10
    prob_class(i) = sum( labelsnew == i-1)/size(labelsnew,1); %MLE za vjerojatnost klase j
end

prob_cond = zeros(784, 10, 2); %prob_cond(i,j,k+1) je MLE za vjerojatnost da je 
                               %i-ti bit slike znamenke jednak k 
                               %(k je 0 ili 1) ako znamenka pripada klasi
                               %j-1

for i=1:784
    for j=1:10
        prob_cond(i, j, 1) = (size(find(X_bin(i, :) == 0 & labelsnew' == j-1), 2))/((prob_class(j)*size(labelsnew, 1)));
        prob_cond(i, j, 2) = (size(find(X_bin(i, :) == 1 & labelsnew' == j-1), 2))/((prob_class(j)*size(labelsnew, 1)));
    end
end
 
suma = 0;


%Klasifikacija

for k=1:10000
    maks = -Inf;
    idx = 1;
    for j=1:10
        for i=1:784
             tmp(i) = prob_cond(i, j, Y_bin(i, k)+1);
        end
         class = log(prob_class(j))+ sum(log(tmp)); %logaritmiramo radi numericke stabilnosti
        if class > maks
            maks = class;
            idx = j;
        end
    end
    if(idx-1 == labels_test(k)) suma = suma +1;
    end
end

acc = (suma/10000)*100;

fprintf('Tocnost je %f %\n',acc);

