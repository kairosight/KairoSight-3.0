clear all
[preData,fps,fname,pname]=tifopen;
dt=1/fps;
data=double(preData);
d = size(data,3);
r = size(data,1);
c = size(data,2);
t = d./fps;
M=zeros(r,c);
vorc = input("apd or cad? ", 's');
for i = 1:r
  for j = 1:c
    zz = data(i,j,:);
    zzz=squeeze(zz);
    if zzz == 0
        continue
    end  
    count = sprintf('r=%d, c=%d',i,j);
    disp(count)
    data2_bwr=detrend(zzz);
    xx=1:length(data2_bwr);
    xnew=1:0.5:length(xx);
    data2_bwr = smooth(data2_bwr);
    data2_bwr_in=interp1(xx,data2_bwr,xnew);
    plot(data2_bwr_in); hold on; plot(data2_bwr_in,'.'); hold off
    %plot(xnew,data2_bwr_in,'.'); hold on; plot(xnew,data2_bwr_in); 
    disp("Select minimum")
    [x1,mini] = ginput(1);
    disp("Select maximum")
    [x2,maxi] = ginput(1);
    if x1 == x2
        apd80t = 0;
        M(i,j) = apd80t;
        continue
    end
    amp = abs(maxi - mini);
    M(i,j) = amp; % saving apd80 value to the correct location in the 2d matrix
  end
end
pcolor(M);axis ij
colormap() %figure out best colormap