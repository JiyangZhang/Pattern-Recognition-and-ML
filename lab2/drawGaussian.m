function Z=drawGaussian(u,v)  % suppose the para are all vectors
% u = [u11,u12; u21,u22; u31,u32; u41,u42]
% v = [v1, v2, v3, v4]
% coordinate range
x=-15:0.05:10;
y=-15:0.05:10;
[X,Y]=meshgrid(x,y);
for i = 1:4
    mu = u(i,:);
    sigma = v(2*i-1:2*i,:);
    DX=sigma(1,1);     %X的方差
    dx=sqrt(DX);
    DY=sigma(2,2);     %Y的方差
    dy=sqrt(DY);
    part1=1/(2*pi*dx*dy);
    p1=-1/2;
    px=(X-mu(1)).^2./DX;
    py=(Y-mu(2)).^2./DY;
    Z=part1*exp(p1*(px+py));  %formula of f(x1, x2)
    surf(X,Y,Z)
    hold on;
end
figure; 
contour(X,Y,Z),title('等高线图');
    %contour(X,Y,Z),title('等高线图')
