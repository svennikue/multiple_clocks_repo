function generateGLM(templateFile, inputFolder)
    % Load glm template file, add in all EVs from input folder, set contrasts
    
    % Load glm template file as cell array with one entry per line
    fid = fopen(templateFile);
    templateText = textscan(fid,'%s','Delimiter','\n');
    templateText = templateText{1};
    fclose(fid);
    
    % Find all EV files in EV folder 
    EVFiles = dir(fullfile(inputFolder,'*.txt'));
    % Collect names of EV files
    EVFiles = {EVFiles.name}';  
    % Enforce alphabetical sorting
    EVFiles = sort(EVFiles);
    % Also find all voxelwise EVs: .nii.gz files
    voxelwiseEVs = dir(fullfile(inputFolder,'*.nii.gz'));
    voxelwiseEVs = {voxelwiseEVs.name}';
    voxelwiseEVs = sort(voxelwiseEVs);
    % Set which EVs are voxelwise, and which are 'normal' - voxelwise go first
    isVoxelwise = [true(size(voxelwiseEVs)); false(size(EVFiles))];
    % And concatenate EV files and voxelwise files
    EVFiles = [voxelwiseEVs; EVFiles];    
    % Get number of EVs
    N = length(EVFiles);
    
    % Check for additional contrasts file in input folder
    if exist(fullfile(inputFolder,'contrasts.json'), 'file') == 2
        % Load additional contrasts from text file
        fileID = fopen(fullfile(inputFolder,'contrasts.json'));
        fileString = textscan(fileID,'%s');
        fileString = strcat(fileString{1}{:});
        % Decode json
        contrasts = jsondecode(fileString);
        % Sort fieldnames alphabetically
        contrastFields = fieldnames(contrasts);
        contrastFields = sort(contrastFields);
    else
        % Set contrasts fields to empty cell
        contrastFields = {};
    end
    
    % Set M: number of contrasts
    M = N + length(contrastFields);
    
    % Make list of name-value pairs that need to be updated with in original template file
    updateEntries = {'evs_orig', num2str(N); 'evs_real', num2str(2*N); 'ncon_orig', num2str(M); 'ncon_real', num2str(M)};
    
    % Run through all entries that need to be updated and replace values by new values
    for currUpdate = 1:size(updateEntries,1)
        % Now start updating the template file. Find line we want to change - this would be the cleanest, but cluster runs R2016a
        %editLoc = find(contains(templateText,updateEntries{currUpdate,1}));
        % So I'll have to write an ugly loop to find where the entry to update occurs in the original lines array
        editLoc = [];
        for currLine = 1:length(templateText)
            if ~isempty(strfind(templateText{currLine},updateEntries{currUpdate,1}))
                editLoc = [editLoc currLine];
            end
        end        
        % Get text that matches the parameter to write
        editOriginal = templateText{editLoc};
        % Deconstruct command into parts: set, name, value - this is clean but doesn't work on matlab 2016
        % editParts = string(textscan(editOriginal,'%s %s %s'));
        % Slightly uglier but backward compatible:
        editParts = textscan(editOriginal,'%s %s %s');
        editParts = [editParts{:}];
        % Update value
        editParts{3} = updateEntries{currUpdate,2};
        % And get updated string
        editUpdated = char(strjoin(editParts));
        % Copy updated string to location of original string
        templateText{editLoc} = editUpdated;
    end
        
    % Find line with "EV 1 title" - that's where we want to start 
    EVLoc = find(ismember(templateText,'# EV 1 title'));
    % Create empty EV string to start with 
    EVString = '';
    % Run through all EVs and construct the EV lines in the fsf file for all of them
    for currEV = 1:N
        % EV entry in fsf file will be different for voxelwise and normal EVs
        if isVoxelwise(currEV)
            % Start by writing the title for this EV 
            currString = ['# EV ' num2str(currEV) ' title' '\n']; 
            currString = [currString 'set fmri(evtitle' num2str(currEV) ') "' strrep(EVFiles{currEV},'.nii.gz','') '"' '\n' '\n'];
            % Write waveform shape
            currString = [currString ...
                '# Basic waveform shape (EV ' num2str(currEV) ')' '\n' ...
                '# 0 : Square' '\n'...
                '# 1 : Sinusoid' '\n'...
                '# 2 : Custom (1 entry per volume)' '\n'...
                '# 3 : Custom (3 column format)' '\n'...
                '# 4 : Interaction' '\n'...
                '# 10 : Empty (all zeros)' '\n'];        
            currString = [currString 'set fmri(shape' num2str(currEV) ') 9' '\n' '\n'];
            % Write convolution
            currString = [currString ...            
                '# Convolution (EV ' num2str(currEV) ')' '\n'...
                '# 0 : None' '\n'...
                '# 1 : Gaussian' '\n'...
                '# 2 : Gamma' '\n'...
                '# 3 : Double-Gamma HRF' '\n'...
                '# 4 : Gamma basis functions' '\n'...
                '# 5 : Sine basis functions' '\n'...
                '# 6 : FIR basis functions' '\n'...
                '# 8 : Alternate Double-Gamma' '\n'];
            currString = [currString 'set fmri(convolve' num2str(currEV) ') 0' '\n' '\n'];
            % Write phase
            currString = [currString '# Convolve phase (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(convolve_phase' num2str(currEV) ') 0' '\n' '\n'];
            % Write temporal filtering
            currString = [currString '# Apply temporal filtering (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(tempfilt_yn' num2str(currEV) ') 0' '\n' '\n'];
            % Write temporal derivative
            currString = [currString '# Add temporal derivative (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(deriv_yn' num2str(currEV) ') 1' '\n' '\n'];
            % Write EV voxelwise image filename
            currString = [currString '# EV ' num2str(currEV) ' voxelwise image filename' '\n'];
            currString = [currString 'set fmri(evs_vox_' num2str(currEV) ') "/Volumes/Scratch_jacobb/derivatives/sub-s01id01/func/EVs/' EVFiles{currEV} '"' '\n' '\n'];
            % Write orthogonalisations
            for currWrt = 0:N
                currString = [currString '# Orthogonalise EV ' num2str(currEV) ' wrt EV ' num2str(currWrt) '\n'];
                currString = [currString 'set fmri(ortho' num2str(currEV) '.' num2str(currWrt) ') 0' '\n' '\n'];
            end            
        else
            % Start by writing the title for this EV 
            currString = ['# EV ' num2str(currEV) ' title' '\n']; 
            currString = [currString 'set fmri(evtitle' num2str(currEV) ') "' strrep(EVFiles{currEV},'.txt','') '"' '\n' '\n'];
            % Write waveform shape
            currString = [currString ...
                '# Basic waveform shape (EV ' num2str(currEV) ')' '\n' ...
                '# 0 : Square' '\n'...
                '# 1 : Sinusoid' '\n'...
                '# 2 : Custom (1 entry per volume)' '\n'...
                '# 3 : Custom (3 column format)' '\n'...
                '# 4 : Interaction' '\n'...
                '# 10 : Empty (all zeros)' '\n'];        
            currString = [currString 'set fmri(shape' num2str(currEV) ') 3' '\n' '\n'];
            % Write convolution
            currString = [currString ...            
                '# Convolution (EV ' num2str(currEV) ')' '\n'...
                '# 0 : None' '\n'...
                '# 1 : Gaussian' '\n'...
                '# 2 : Gamma' '\n'...
                '# 3 : Double-Gamma HRF' '\n'...
                '# 4 : Gamma basis functions' '\n'...
                '# 5 : Sine basis functions' '\n'...
                '# 6 : FIR basis functions' '\n'...
                '# 8 : Alternate Double-Gamma' '\n'];
            currString = [currString 'set fmri(convolve' num2str(currEV) ') 2' '\n' '\n'];
            % Write phase
            currString = [currString '# Convolve phase (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(convolve_phase' num2str(currEV) ') 0' '\n' '\n'];
            % Write temporal filtering
            currString = [currString '# Apply temporal filtering (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(tempfilt_yn' num2str(currEV) ') 1' '\n' '\n'];
            % Write temporal derivative
            currString = [currString '# Add temporal derivative (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(deriv_yn' num2str(currEV) ') 1' '\n' '\n'];
            % Write EV file
            currString = [currString '# Custom EV file (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(custom' num2str(currEV) ') "/Volumes/Scratch_jacobb/derivatives/sub-s01id01/func/EVs/' EVFiles{currEV} '"' '\n' '\n'];
            % Write gamma sigma
            currString = [currString '# Gamma sigma (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(gammasigma' num2str(currEV) ') 3' '\n' '\n'];
            % Write gamma delay 
            currString = [currString '# Gamma delay (EV ' num2str(currEV) ')' '\n'];
            currString = [currString 'set fmri(gammadelay' num2str(currEV) ') 6' '\n' '\n'];
            % Write orthogonalisations
            for currWrt = 0:N
                currString = [currString '# Orthogonalise EV ' num2str(currEV) ' wrt EV ' num2str(currWrt) '\n'];
                currString = [currString 'set fmri(ortho' num2str(currEV) '.' num2str(currWrt) ') 0' '\n' '\n'];
            end
        end
        % Finally: add current string to EV string
        EVString = [EVString currString];
    end
    % Find where feat directory part ends
    EVEndLoc = find(ismember(templateText,'# Contrast & F-tests mode'));
    % Remove everything between EVLoc and EVEndLoc
    templateText = templateText([1:EVLoc EVEndLoc:length(templateText)]);
    % Write the EV string to the EV location (for testing: just above EV location)
    templateText{EVLoc} = EVString;
    
    % Now do something similar for contrasts. First find where to start
    contrastLoc = find(ismember(templateText,'# Display images for contrast_real 1'));
    % Create empty constrast string
    contrastString = '';
    % Run through all EVs and make a contrast for each of them, having a 1 for that EV and zeros everywhere else. First: 'real' (includes derivatives)
    for currContrast = 1:N
        % Start by writing whether to display images for this contrast
        currString = ['# Display images for contrast_real ' num2str(currContrast) '\n'];
        currString = [currString 'set fmri(conpic_real.' num2str(currContrast) ') 1' '\n' '\n'];
        % Write title
        currString = [currString '# Title for contrast_real ' num2str(currContrast) '\n'];
        currString = [currString 'set fmri(conname_real.' num2str(currContrast) ') "' strrep(EVFiles{currContrast},'.txt','') '"' '\n' '\n'];
        % Write contrast vector 
        for currElement = 1:(2*N)
            currString = [currString '# Real contrast_real vector ' num2str(currContrast) ' element ' num2str(currElement) '\n'];
            % Only write a 1 for the element corresponding to the current EV (skipping the temporal derivatives)
            if currElement == (currContrast-1)*2+1
                currString = [currString 'set fmri(con_real' num2str(currContrast) '.' num2str(currElement) ') 1' '\n' '\n'];
            else
                currString = [currString 'set fmri(con_real' num2str(currContrast) '.' num2str(currElement) ') 0' '\n' '\n'];
            end
        end
        % Add current string to contrast string
        contrastString = [contrastString currString];
    end
    % Now run through additional contrasts and create contrast lines for those
    for currAdditionalContrast = 1:length(contrastFields)
        % Set current contrast: starts of at N
        currContrast = N + currAdditionalContrast;
        % Start by writing whether to display images for this contrast
        currString = ['# Display images for contrast_real ' num2str(currContrast) '\n'];
        currString = [currString 'set fmri(conpic_real.' num2str(currContrast) ') 1' '\n' '\n'];
        % Write title
        currString = [currString '# Title for contrast_real ' num2str(currContrast) '\n'];
        currString = [currString 'set fmri(conname_real.' num2str(currContrast) ') "' contrastFields{currAdditionalContrast} '"' '\n' '\n'];
        % Write contrast vector 
        for currElement = 1:(2*N)
            currString = [currString '# Real contrast_real vector ' num2str(currContrast) ' element ' num2str(currElement) '\n'];
            % Write the contrast matrix value for the current EV (skipping the temporal derivatives)
            if contrasts.(contrastFields{currAdditionalContrast})(ceil(currElement/2)) ~= 0 && mod(currElement,2)==1
                currString = [currString 'set fmri(con_real' num2str(currContrast) '.' num2str(currElement) ') ' num2str(contrasts.(contrastFields{currAdditionalContrast})(ceil(currElement/2))) '\n' '\n'];
            else
                currString = [currString 'set fmri(con_real' num2str(currContrast) '.' num2str(currElement) ') 0' '\n' '\n'];
            end
        end
        % Add current string to contrast string
        contrastString = [contrastString currString];
    end
    
    % Run through all EVs and make a contrast for each of them, having a 1 for that EV and zeros everywhere else. Second: 'orig' (only the ones specified by user)
    for currContrast = 1:N
        % Start by writing whether to display images for this contrast
        currString = ['# Display images for contrast_orig ' num2str(currContrast) '\n'];
        currString = [currString 'set fmri(conpic_orig.' num2str(currContrast) ') 1' '\n' '\n'];
        % Write title
        currString = [currString '# Title for contrast_orig ' num2str(currContrast) '\n'];
        currString = [currString 'set fmri(conname_orig.' num2str(currContrast) ') "' strrep(EVFiles{currContrast},'.txt','') '"' '\n' '\n'];
        % Write contrast vector 
        for currElement = 1:N
            currString = [currString '# Real contrast_orig vector ' num2str(currContrast) ' element ' num2str(currElement) '\n'];
            % Only write a 1 for the element corresponding to the current EV
            if currElement == currContrast
                currString = [currString 'set fmri(con_orig' num2str(currContrast) '.' num2str(currElement) ') 1' '\n' '\n'];
            else
                currString = [currString 'set fmri(con_orig' num2str(currContrast) '.' num2str(currElement) ') 0' '\n' '\n'];
            end
        end
        % Add current string to contrast string
        contrastString = [contrastString currString];
    end    
    % And again run through additional contrasts and create contrast lines for those
    for currAdditionalContrast = 1:length(contrastFields)
        % Set current contrast: starts of at N
        currContrast = N + currAdditionalContrast;
        % Start by writing whether to display images for this contrast
        currString = ['# Display images for contrast_orig ' num2str(currContrast) '\n'];
        currString = [currString 'set fmri(conpic_orig.' num2str(currContrast) ') 1' '\n' '\n'];
        % Write title
        currString = [currString '# Title for contrast_orig ' num2str(currContrast) '\n'];
        currString = [currString 'set fmri(conname_orig.' num2str(currContrast) ') "' contrastFields{currAdditionalContrast} '"' '\n' '\n'];
        % Write contrast vector 
        for currElement = 1:N
            currString = [currString '# Real contrast_orig vector ' num2str(currContrast) ' element ' num2str(currElement) '\n'];
            % Write the contrast matrix value for the current EV
            if contrasts.(contrastFields{currAdditionalContrast})(currElement) ~= 0
                currString = [currString 'set fmri(con_orig' num2str(currContrast) '.' num2str(currElement) ') ' num2str(contrasts.(contrastFields{currAdditionalContrast})(currElement)) '\n' '\n'];
            else
                currString = [currString 'set fmri(con_orig' num2str(currContrast) '.' num2str(currElement) ') 0' '\n' '\n'];
            end
        end
        % Add current string to contrast string
        contrastString = [contrastString currString];
    end        
        
    % Write contrast masking
    contrastString = [contrastString '# Contrast masking - use >0 instead of thresholding?' '\n'];
    contrastString = [contrastString 'set fmri(conmask_zerothresh_yn) 0' '\n' '\n'];
    % Run through all contrasts to set contrast masking - currently all 0
    for currContrast1 = 1:M
        for currContrast2 = 1:M
            if currContrast1 ~= currContrast2
                contrastString = [contrastString '# Mask real contrast/F-test ' num2str(currContrast1) ' with real contrast/F-test ' num2str(currContrast2) '?' '\n'];
                contrastString = [contrastString 'set fmri(conmask' num2str(currContrast1) '_' num2str(currContrast2) ') 0' '\n' '\n'];
            end
        end
    end
    % Write to do contrast masking at all
    contrastString = [contrastString '# Do contrast masking at all?' '\n'];
    contrastString = [contrastString 'set fmri(conmask1_1) 0' '\n' '\n'];
    % Find where feat contrast part ends
    contrastEndLoc = find(ismember(templateText,'##########################################################'));
    % Remove everything between contrastLoc and contrastEndLoc - 1
    templateText = templateText([1:contrastLoc contrastEndLoc:length(templateText)]);    
    % Write the contrast string to the contrast location (for testing: just above contrast location)
    templateText{contrastLoc} = contrastString;
    
    % Join the whole string into one big output string
    outputString = strjoin(templateText,'\n');
    % Write output to new file
    fid = fopen(strrep(templateFile,'.fsf','_full.fsf'),'wt');
    fprintf(fid, strrep(outputString, '%', '%%')); % Need to replace % by %% because Matlabs likes to interpret it
    fclose(fid);        
end