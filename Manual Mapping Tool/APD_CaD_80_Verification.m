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
calc = input("30 or 80? ");
for i = 1:r
  for j = 1:c
    zz = data(i,j,:);
    zzz=squeeze(zz);
    if zzz == 0
        continue
    end  
    count = sprintf('r=%d, c=%d',i,j);
    disp(count)
    if vorc == "apd"
        zzz_inv=zzz.*-1;
        data2_bwr=detrend(zzz_inv);
    elseif vorc == "cad"
        data2_bwr=detrend(zzz);
    end
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
    disp("Select dv/dt")
    [x3,dvdt] = ginput(1);
    disp("Select end of action potential")
    [x4,fin] = ginput(1);
    amp = maxi - mini;
    if calc == 30
        percnum = amp*.7 + mini;
    elseif calc == 80
        percnum = amp*0.2 + mini;
    end   
    x2r = round(x2);
    x4r = round(x4);
    repol=data2_bwr_in(x2r:x4r);
    plot(repol)
    min_x=1000000;
    for n=1:(length(repol)-1)
        xx(n)=abs(percnum-repol(n));
        if(xx(n)<min_x)
            min_x=xx(n);
            x_value=n;
        end
    end
    x_80perc = x_value+x2-1;
    apd80 = (x_80perc - x3)/2;
    apd80t = apd80*dt*1000; %converting to time (ms)
    M(i,j) = apd80t; % saving apd80 value to the correct location in the 2d matrix
  end
end
pcolor(M);axis ij
colormap() %figure out best colormap