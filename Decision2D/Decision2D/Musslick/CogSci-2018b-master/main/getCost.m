function controlCost = getCost(controlIntensity, costParam)

controlCost = exp(controlIntensity * costParam);

end