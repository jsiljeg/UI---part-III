load('mnistd.mat');
X=X';
for i=1:10 
    T{i} = X(:, labels == i-1);  %Skupovi Ti
end

for i=1:10
[U{i}, ~, ~] = svd(T{i});   %Ovdje vršimo SVD dekompoziciju i dobijemo skupove Ui (singularni vektori od Ti)
end

m = 10; %mogu i drukèiji m-ovi

for i=1:10
Um{i} = U{i}(:,1:m)*U{i}(:,1:m)';  %Uzimamo prvih m sing. vektora
end

suma = 0;

%Klasifikacija testnih znamenki

for i=1:10000
for j=1:10 dist(j) = norm(Y(:,i)-Um{j}*Y(:,i)); end  %Racunamo reziduale
[~,idx] = min(dist);
if(idx-1 == labels_test(i)) suma = suma+1;   %Usporedba s pravim labelima
end
end
fprintf ('Postotak iznosi %f %', suma/100)

