function RDM = RSAGenerateRDMs(Y, SPM, conditions, distType)
    % This is called by rsa.runSearchlight and calculates RDMs for given raw data Y (but selected within searchlight!)
    
    % Previously I used my own variant of noiseNormalizeBeta called noiseNormalizeBetaJB
    % To make sure I won't have any biases from the mean signal, I removed the mean from regressors and data
    % I have now moved the manipulations of the design matrix to RSARunSearchlights(Vol) which makes more sense
    % So there is essentially no reason to still use noiseNormalizeBetaJB - except if you want to demean data
    % (But with demenead regressors, it shouldn't actually matter whether the data is demeaned or not)
    % One more reason to use noiseNormalizeBetaJB: it doesn't high pass filter Y (which was already done in preproc)
    
    % Do GLM and get prewhitened beta weights    
    %[Bwhitened,~,~,B] = noiseNormalizeBetaJB(Y,SPM);     
    Bwhitened = rsa.spm.noiseNormalizeBeta(Y,SPM);
    B = real(Bwhitened);
    % Only keep selected conditions for RDM (get rid of nuisance regressors and derivatives)
    B = B(conditions,:);
    % Calculate distance, returns a vector
    RDM = pdist(B,distType);
    % And turn into list
    RDM = RDM(:);
end