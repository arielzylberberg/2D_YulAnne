function [c_data, offset_data, sample_data] = packEstimationResults(c1, c2, c1_hat, c2_hat, offset1, offset2, offset1_hat, offset2_hat, estimatedCostFunction1, estimatedCostFunction2, estimatedControlSignals1, estimatedControlSignals2, controlSignalSpace)


    c_data.c1 = c1;
    c_data.c2 = c2;
    c_data.c1_hat = c1_hat;
    c_data.c2_hat = c2_hat;

    offset_data.offset1 = offset1;
    offset_data.offset2 = offset2;
    offset_data.offset1_hat = offset1_hat;
    offset_data.offset2_hat = offset2_hat;

    sample_data.controlSignalSpace = controlSignalSpace;

    sample_data.estimatedCostFunction1 = estimatedCostFunction1;
    sample_data.estimatedCostFunction2 = estimatedCostFunction2;

    sample_data.estimatedControlSignals1 = estimatedControlSignals1;
    sample_data.estimatedControlSignals2 = estimatedControlSignals2;
    
end