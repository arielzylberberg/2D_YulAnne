function x1 = updateLeakyIntegrator(x0,dx,lambda,dt,varargin)
% x1 = updateLeakyIntegrator(x0,dx,lambda,dt,varargin) implements an update
% of a leaky integrator with time constant lambda, from x0 to x1 over time
% step, dt, given input dx. If varargin, 'reset',true, then x0 is overriden
% to the value of 'resetVal' [default 0].

% Jan 2019 mns wrote it. 

pSet = inputParser;
addParameter(pSet,'reset',false,@islogical);
addParameter(pSet,'resetVal',0,@isscalar);

parse(pSet,varargin{:});
%%
if pSet.Results.reset
    x0 = pSet.Results.resetVal;
end
x1 = x0 + ((dx/lambda) - (1/lambda)*x0)*dt;

