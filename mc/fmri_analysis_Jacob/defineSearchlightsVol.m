function defineSearchlightsVol(searchlightFolder)
    % Define searchlights within a volumetric ROI in MNI space
    % Only include voxels within a cortex + hippocampus mask,
    % by adding cortex and left and right hippocampus in the Harvard-Oxford
    % subcortical structural atlas together at threshold 10
    
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
    % And also add the RSA toolbox to path
    rsaPath = fullfile(scratchDir,'matlab','rsatoolbox');
    % Add RSA folder with all subdirectories to Matlab
    addpath(genpath(rsaPath));                

    % Base directory of mask
    maskDir = fullfile(searchlightFolder);

    % Read mask for RSA
    disp(['################################## ' newline ' Reading mask ' newline ' ##################################']);
    rsaMask = spm_vol(fullfile(searchlightFolder, 'all_brain_mask.nii')); % Was mask_corthippo.nii
    rsaMask.data = spm_read_vols(rsaMask);
    rsaMask.sphere=[10,100];

    % Get searchlights.
    disp(['################################## ' newline ' Defining searchlights ' newline ' ##################################']);
    Searchlights = rsa.defineSearchlight({rsaMask},rsaMask);

    % And store result
    disp(['################################## ' newline ' Searchlights successfully stored! ' newline ' ##################################']);
    save(fullfile(searchlightFolder,'SL_allbrain100.mat'),'-struct','Searchlights'); % Was SL_vol100.mat

end
