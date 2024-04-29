% this is tim's lesson in regressions: matrix multiplication method.

for i=1:100
x1=randn(100,1);
x2=x1+randn(100,1);
x=[x1-mean(x1) x2-mean(x2)];

y=randn(100,1); y=y-mean(y);

betap(:,i)=pinv(x)*y;
betamanual(:,i)=inv(x'*x)*x'*y;

betaridge(:,i)=inv(x'*x + 1000*eye(size(x,2)))*x'*y;
end

scatter(betamanual(1,:),betamanual(2,:))


