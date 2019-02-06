function [myU,myS,myV] = mySVD(A)
[M,N]=size(A);
[myV,myS] = eig(A'*A);
myV = fliplr(myV);
myS = rot90(myS,2);
myS = sqrt(myS(1:M,1:N));
myU = A*myV*pinv(myS);
