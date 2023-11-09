function RSACrossBlockRDM(inputFolder, subjectTag, outputFolder)
    % To avoid time influencing the RDM: take dissimilarity between the same pair of conditions across runs
    % RDM(cond_i, cond_j) = (RDM_old(cond_i_block1, cond_j_block2) + RDM_old(cond_j_block_1, cond_i_block_2))/2
    
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
    % Set path of Imaging folder containing scripts
    imagingPath = fullfile(scratchDir,'matlab','imaging');
    % Add all imaging folder with all subdirectories to Matlab
    addpath(genpath(imagingPath));    
    % Add the RSA toolbox to path
    rsaPath = fullfile(scratchDir,'matlab','rsatoolbox');
    % Add RSA folder with all subdirectories to Matlab
    addpath(genpath(rsaPath));        
    
    % Get FSL path
    fsldir = getenv('FSLDIR');
    
    % Base directory of fMRI data
    derivDir = fullfile(scratchDir,'derivatives',['sub-' subjectTag]);

    % Load SPM structure of GLM
    disp('Loading GLM results...');
    SPM = load(fullfile(inputFolder,'SPM.mat')); SPM = SPM.SPM;
       
    % The big RDM created previously contains all conditions. Retrieve which is which
    stimConditions = [];
    relConditions = [];
    block1Conditions = [];
    block2Conditions = [];
    % Find the conditions in each block
    for currRun = 1:length(SPM.Sess)
        % Find catch and response regressors: we don't want to include those
        catchAndResponse = contains({SPM.Sess(currRun).Fc.name},'Catch') | contains({SPM.Sess(currRun).Fc.name},'Response');
        % Find all conditions for stimuli
        stimConditions = [stimConditions, contains({SPM.Sess(currRun).Fc.name},'Stim') & ~catchAndResponse];
        % Find all conditions for relations
        relConditions = [relConditions, contains({SPM.Sess(currRun).Fc.name},'Rel') & ~catchAndResponse];
        % Find all conditions for block 1
        block1Conditions = [block1Conditions, contains({SPM.Sess(currRun).Fc.name},'Block1') & ~catchAndResponse];
        % Find all conditions for block 2
        block2Conditions = [block2Conditions, contains({SPM.Sess(currRun).Fc.name},'Block2') & ~catchAndResponse];
    end
    % Find all conditions overall, excluding nuisance regressors
    allConditions = stimConditions | relConditions | block1Conditions | block2Conditions;
    % And remove all nuisance regressors from the condition arrays
    stimConditions = stimConditions(allConditions);
    relConditions = relConditions(allConditions);
    block1Conditions = block1Conditions(allConditions);
    block2Conditions = block2Conditions(allConditions);
    % Also reduce allConditions length
    allConditions = allConditions(allConditions);
    % Calculate number of entries
    entries = length(allConditions) * (length(allConditions)-1) /2;
    
    % Load RDM nii file info
    V = spm_vol(fullfile(outputFolder,'RDM.nii'));
    % Verify that the number of volumes in that file is equal to the number of entries
    disp(['Is expected number of entries (' num2str(entries) ') equal to discovered number of entries (' num2str(length(V)) ')? ' num2str(entries == length(V)) '.'])
    % And load actual data. Use read_avw as it's much faster than spm_read_vols
    disp('Loading data...');
    % Y = spm_read_vols(V);
    Y = read_avw(fullfile(outputFolder,'RDM.nii'));
    
    % Get new RDM size
    newSize = sum(relConditions & block1Conditions);
    % And new number of entries
    newEntries = newSize * (newSize - 1) / 2;
    
    % Find all entries in flattened full RDM that correspond to rel condition, block 1 and block 2
    relEntries = zeros(length(allConditions));
    relEntries(relConditions & block1Conditions, relConditions & block2Conditions) = 1;
    relEntries = squareform(max(relEntries,relEntries'));
    % Create rel condition matrix
    rel = zeros([size(Y,1) size(Y,2) size(Y,3) newEntries]);
    % Also store diagonal, which isn't zero anymore
    relDiag = zeros([size(Y,1) size(Y,2) size(Y,3) newSize]);
    % Run through 3D nii values
    disp('Building new rel RDM...');
    for i = 1:size(Y,1)
        for j = 1:size(Y,2)
            for k = 1:size(Y,3)
                % Get all RDM values corresponding to the rel condition in block 1 vs block 2
                currRDM = reshape(Y(i,j,k,logical(relEntries)),newSize,newSize);
                % Average distances i,j and j,i
                currRDM = (currRDM + currRDM')/2;
                % Store diagonal
                relDiag(i,j,k,:) = diag(currRDM);
                % Remove diagonal; can't do -diag(diag()) because of NaNs
                currRDM(1:newSize+1:end) = 0;
                % And store flattened version
                rel(i,j,k,:) = squareform(currRDM);
            end
        end
    end
    % And store results
    save4Dnii(rel,V,fullfile(outputFolder,'relRDM.nii'));
    save4Dnii(relDiag,V,fullfile(outputFolder,'relRDMdiag.nii'));    
    
    % The stim condition does not necessarily exist (depends on settings during EV generation)
    if sum(stimConditions)>0
        % Find all entries in flattened full RDM that correspond to stim condition, block 1 and block 2
        stimEntries = zeros(length(allConditions));
        stimEntries(stimConditions & block1Conditions, stimConditions & block2Conditions) = 1;
        stimEntries = squareform(max(stimEntries,stimEntries'));
        % Create rel condition matrix
        stim = zeros([size(Y,1) size(Y,2) size(Y,3) newEntries]);
        % Also store diagonal, which isn't zero anymore
        stimDiag = zeros([size(Y,1) size(Y,2) size(Y,3) newSize]);
        % Run through 3D nii values
        disp('Building new stim RDM...');
        for i = 1:size(Y,1)
            for j = 1:size(Y,2)
                for k = 1:size(Y,3)
                    % Get all RDM values corresponding to the stim condition in block 1 vs block 2
                    currRDM = reshape(Y(i,j,k,logical(stimEntries)),newSize,newSize);
                    % Average distances i,j and j,i
                    currRDM = (currRDM + currRDM')/2;
                    % Store diagonal
                    stimDiag(i,j,k,:) = diag(currRDM);
                    % Remove diagonal
                    currRDM(1:newSize+1:end) = 0;
                    % And store flattened version
                    stim(i,j,k,:) = squareform(currRDM);
                end
            end
        end
        % And store results
        save4Dnii(stim,V,fullfile(outputFolder,'stimRDM.nii'));
        save4Dnii(stimDiag,V,fullfile(outputFolder,'stimRDMdiag.nii'));  
    end
end