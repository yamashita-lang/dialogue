function ssinds = getResampInd10fold(samplenum, divsample)

if ~exist('divsample', 'var')
    divnum = 50;
else
    divnum = samplenum/divsample;
end

% get cross validation samples
cvFrac = 0.1;
randSeed = 1234;
rand('twister',randSeed);
randn('state',randSeed);

zs=floor(samplenum/divnum);
zr = reshape(1:zs*divnum, zs, divnum);

ssinds = [];
a=randperm(divnum);
bin = round(divnum*cvFrac);
for ii=1:10
  regInd = zr(:,a(1+bin*(ii-1):bin*ii));
  regInd = regInd(:)';
  trnInd = setdiff(1:samplenum, [regInd]);
  ssinds(ii).regInd = regInd;
  ssinds(ii).trnInd = trnInd;
end

