function rsync(op, pth)

local = pth;
remote = fullfile('yul@pat.shadlen.zi.columbia.edu:~/s/Decision2D/Decision2D', pth);

switch op
    case 'push'
        src = local;
        dst = remote;
        
    case 'pull'
        src = remote;
        dst = local;
end

cmd = sprintf('rsync -avz -e ssh %s %s', src, dst); 
system_prudent(cmd);