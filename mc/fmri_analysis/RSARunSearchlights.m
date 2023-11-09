function RSARunSearchlights(inputFolder, subjectTag, outputFolder, isVol)
    % Run GLM in SPM with EVs from EVfile separately for each searchlight, and calculate resulting RDMs
    
    % Optional argument: is this for volumetric or surface analysis?
    if nargin < 4
        % By default: set surface. Default not consistent with RSAGenerateGLM for historical reasons
        isVol = false;
    end
    
    % Set default home dir for execution on cluster
    homeDir = '/home/fs0/jacobb';
    if ~exist(homeDir,'dir')
        % If not called on cluster, but working on laptop connected to server
        homeDir = '/Volumes/jacobb';
    end
    if ~exist(homeDir,'dir')
        % If not called on laptop, but on mac desktop connected to server
        homeDir = '/Users/jacobb/Documents/ServerHome';
    end
    
    % Set default home dir for execution on cluster
    scratchDir = '/home/fs0/jacobb/scratch';
    if ~exist(scratchDir,'dir')
        % If not called on cluster, but working on laptop connected to server
        scratchDir = '/Volumes/Scratch_jacobb';
    end
    if ~exist(scratchDir,'dir')
        % If not called on laptop, but on mac desktop connected to server
        scratchDir = '/Users/jacobb/Documents/ServerHome/scratch';
    end
    
    % Set path of SPM installation
    spmPath = fullfile(scratchDir,'matlab','spm12');
    % Add SPM and all subdirectories to Matlab
    addpath(genpath(spmPath));    
    % Add the RSA toolbox to path
    rsaPath = fullfile(scratchDir,'matlab','rsatoolbox');
    % Add all imaging folder with all subdirectories to Matlab
    addpath(genpath(rsaPath));        
    
    % Base directory of fMRI data
    derivDir = fullfile(scratchDir,'derivatives',['sub-' subjectTag]);

    % Load searchlight definitions
    disp('Loading searchlights...');
    if isVol
        searchlights = load(fullfile(homeDir,'Analysis','Masks','VolumetricSearchlights','SL_vol100.mat'));
    else
        searchlights = load(fullfile(derivDir,'func','searchlight','SL_surf.mat'));
    end
    
    % Load SPM structure of GLM
    disp('Loading GLM results...');
    SPM = load(fullfile(inputFolder,'SPM.mat')); SPM = SPM.SPM;
    
    % Set type of distance as used in RDM
    distType = 'correlation';
    
    % Select all GLM regressors to be included as conditions in the RDMs
    conditions = false(1,size(SPM.xX.xKXs.X,2));
    % Collect conditions throughout runs
    for currRun = 1:length(SPM.Sess)
        % Get all relevant regressors (without motion and nuisance regressors): those with Stim or Rel in the name
        realRegressors = find((contains({SPM.Sess(currRun).Fc.name},'Stim') | contains({SPM.Sess(currRun).Fc.name},'Rel')) ...
            & (~contains({SPM.Sess(currRun).Fc.name},'Catch') & ~contains({SPM.Sess(currRun).Fc.name},'Response')));
        % And for all of them: set a 1 in the entry corresponding to the actual regressor (not the derivative)
        for currRegressorIdx = 1:length(realRegressors)
            currRegressor = realRegressors(currRegressorIdx);
            conditions(SPM.Sess(currRun).col(SPM.Sess(currRun).Fc(currRegressor).i(1))) = true;
        end
    end
    % Count number of RDM entries
    entries = sum(conditions)*(sum(conditions)-1)/2;

    % The below might actually be dumb. You probably don't want to filter motion outliers and mean regressor?
    % But even without this filtering: SPM seems to apply a filter to the design matrix!
    % Optional: high pass filter all EVs. Sigma = cutoff time in seconds / 2 / TR
%     SPM.xX.xKXs.X = highPassFilter(SPM.xX.xKXs.X,40.5,false);
%     
%     % Instead of using SPM's design matrix with mean regressor, I'll demean data and regressors manually
%     % Remove the last column of the design matrix, corresponding to the mean regressor
%     % Careful: many of the remaining fields will still reflect the old number of regressors
%     % This is fine as long as they are not used in fitting the GLM
%     SPM.xX.xKXs.X = SPM.xX.xKXs.X(:,1:(end-1));  
%     % Demean all EVs
%     SPM.xX.xKXs.X = SPM.xX.xKXs.X - repmat(mean(SPM.xX.xKXs.X,1),[size(SPM.xX.xKXs.X,1),1]);
%     % Optional, nicer for plotting: normalise EVs
%     SPM.xX.xKXs.X = SPM.xX.xKXs.X ./ repmat(max(abs(SPM.xX.xKXs.X),[],1), [size(SPM.xX.xKXs.X,1),1]);
%     % Precalculate inv(X'*X)*X'*Y
%     SPM.xX.pKX = (SPM.xX.xKXs.X' * SPM.xX.xKXs.X) \ SPM.xX.xKXs.X'; 
%     
    % Create output files: upper/lower triangle of RDM, same filename but with ',n' at the end to indicate n-th slice
    outFiles = cell(entries,1);
    for currEntry = 1:entries
        outFiles{currEntry} = fullfile(outputFolder,['RDM.nii',',',num2str(currEntry)]);
    end
    
    % Create analysis name: name for saving temporary files
    analysisName = strsplit(outputFolder,filesep);
    % Analysis name has subject tag and RSA number
    analysisName = [subjectTag '_' analysisName{end-1}];
    
    % And run the searchlight, with a function handle to specify how RDMs are calculated
    disp(['Calculating RDMs for each searchlight for analysis ' analysisName]);
    % Alon has the analysis name as the fifth argument, but that doesn't match the function description
    rsa.runSearchlight(searchlights,SPM.xY.VY, outFiles, @RSAGenerateRDMs,'optionalParams',{SPM,conditions,distType});
    disp(['Success! Files saved in ' outFiles{1}]);
end