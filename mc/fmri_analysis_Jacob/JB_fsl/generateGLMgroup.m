function generateGLMgroup(templateFile, subjectDir, subjectList, subjectContrasts)
    % Load glm template file, add in all EVs from input folder, set contrasts
    
    % Load glm template file as cell array with one entry per line
    fid = fopen(templateFile);
    templateText = textscan(fid,'%s','Delimiter','\n');
    templateText = templateText{1};
    fclose(fid);
    
    % Extract all subjects from subjectList, containing IDs separated by spaces
    tags = strsplit(subjectList);
    % Get total number of subjects
    N = length(tags);
    
    % Get struct with all EVs for these subjects
    EVs = generateEVsGroup(subjectDir,subjectList,false);
    % Get names of EV fields
    EVnames = fieldnames(EVs);
    % Store the number of group level EVs
    M = length(EVnames);
    
    % Make list of name-value pairs that need to be updated with in original template file
    updateEntries = {'npts', num2str(N); 'multiple', num2str(N); 'ncopeinputs', num2str(subjectContrasts); 'evs_orig', num2str(M); 'evs_real', num2str(M); 'ncon_real', num2str(M);};
    
    % Run through all entries that need to be updated and replace values by new values
    for currUpdate = 1:size(updateEntries,1)
        % Now start updating the template file. Fine line we want to change - this would be the cleanest, but cluster runs R2016a
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
       
    % Find line with "# Use lower-level cope 1 for higher-level analysis", start writing copes
    copeLoc = find(ismember(templateText,'# Use lower-level cope 1 for higher-level analysis'));
    % Create empty feat directory string to start with 
    copeString = '';
    % Run through all subject tags and construct the feat directory lines in the fsf file for all of them
    for currCope = 1:subjectContrasts
        % Write title for current feat directory
        currString = ['# Use lower-level cope ' num2str(currCope) ' for higher-level analysis' '\n']; 
        currString = [currString 'set fmri(copeinput.' num2str(currCope) ') 1' '\n' '\n'];        
        % Add current string to feat directory string
        copeString = [copeString currString];
    end
    % Find where cope part ends
    copeEndLoc = find(ismember(templateText,'# 4D AVW data or FEAT directory (1)'));
    % Remove everything from copeLoc + 1 to copeEndLoc - 1
    templateText = templateText([1:(copeLoc) (copeEndLoc):length(templateText)]);
    % Write the feat directory string to the feat directory location
    templateText{copeLoc} = copeString;        
    
    % Find line with "# 4D AVW data or FEAT directory (1)" - that's where we want to start 
    featLoc = find(ismember(templateText,'# 4D AVW data or FEAT directory (1)'));
    % Copy the line that comes directly after that, to replace the subject tag later
    origFeatLine = templateText{featLoc+1};
    % Create empty feat directory string to start with 
    featString = '';
    % Run through all subject tags and construct the feat directory lines in the fsf file for all of them
    for currTag = 1:N
        % Write title for current feat directory
        currString = ['# 4D AVW data or FEAT directory (' num2str(currTag) ')' '\n']; 
        % Write filename for current feat directory by replacing subject tag
        currString = [currString strrep(strrep(origFeatLine,'(1)',['(' num2str(currTag) ')']),'s01id01',tags{currTag}) '\n' '\n'];        
        % Add current string to feat directory string
        featString = [featString currString];
    end
    % Find where feat directory part ends
    featEndLoc = find(ismember(templateText,'# Add confound EVs text file'));
    % Remove everything between featLoc + 1 and featEndLoc - 1
    templateText = templateText([1:(featLoc) (featEndLoc):length(templateText)]);
    % Write the feat directory string to the feat directory location
    templateText{featLoc} = featString;
    
    % Now write EVs as generated earlier, stored in Matlab struct
    EVLoc = find(ismember(templateText,'# EV 1 title'));
    % Create empty EV string to start with 
    EVString = '';
    % Run through all EVs and construct the EV lines in the fsf file for all of them
    for currEV = 1:M
        % Start by writing the title for this EV 
        currString = ['# EV ' num2str(currEV) ' title' '\n']; 
        currString = [currString 'set fmri(evtitle' num2str(currEV) ') "' EVnames{currEV} '"' '\n' '\n'];
        % Write waveform shape
        currString = [currString ...
            '# Basic waveform shape (EV ' num2str(currEV) ')' '\n' ...
            '# 0 : Square' '\n'...
            '# 1 : Sinusoid' '\n'...
            '# 2 : Custom (1 entry per volume)' '\n'...
            '# 3 : Custom (3 column format)' '\n'...
            '# 4 : Interaction' '\n'...
            '# 10 : Empty (all zeros)' '\n'];        
        currString = [currString 'set fmri(shape' num2str(currEV) ') 2' '\n' '\n'];
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
        currString = [currString 'set fmri(deriv_yn' num2str(currEV) ') 0' '\n' '\n'];
        % Write EV file
        currString = [currString '# Custom EV file (EV ' num2str(currEV) ')' '\n'];
        currString = [currString 'set fmri(custom' num2str(currEV) ') "dummy"' '\n' '\n'];
        % Write orthogonalisations
        for currWrt = 0:M
            currString = [currString '# Orthogonalise EV ' num2str(currEV) ' wrt EV ' num2str(currWrt) '\n'];
            currString = [currString 'set fmri(ortho' num2str(currEV) '.' num2str(currWrt) ') 0' '\n' '\n'];
        end
        % Write higher level values, taken from EV struct
        for currTag = 1:N
            currString = [currString '# Higher-level EV value for EV ' num2str(currEV) ' and input ' num2str(currTag) '\n'];
            currString = [currString 'set fmri(evg' num2str(currTag) '.' num2str(currEV) ') ' num2str(EVs.(EVnames{currEV})(currTag)) '\n' '\n'];
        end
        % Finally: add current string to EV string
        EVString = [EVString currString];
    end
    % Find where EV part ends
    EVEndLoc = find(ismember(templateText,'# Setup Orthogonalisation at higher level? '));
    % Remove everything between EVLoc and EVEndLoc
    templateText = templateText([1:EVLoc EVEndLoc:length(templateText)]);
    % Write the EV string to the EV location (for testing: just above EV location)
    templateText{EVLoc} = EVString;    
      
    % Write group memberships
    groupLoc = find(ismember(templateText,'# Group membership for input 1'));
    % Create empty EV string
    groupString = '';
    % Run through all subjects and set their group
    for currTag = 1:N
        % Write group membership of current subject
        currString = ['# Group membership for input ' num2str(currTag) '\n'];
        currString = [currString 'set fmri(groupmem.' num2str(currTag) ') 1' '\n' '\n'];
        % Add current string to EV string
        groupString = [groupString currString];
    end    
    % Find where group part ends    
    groupEndLoc = find(ismember(templateText,'# Contrast & F-tests mode'));
    % Remove everything between groupLoc and groupEndLoc
    templateText = templateText([1:(groupLoc) (groupEndLoc):length(templateText)]);
    % Write the group string to the group location
    templateText{groupLoc} = groupString;    

    % Write contrast: one contrast for each EV
    contrastLoc = find(ismember(templateText,'# Display images for contrast_real 1'));
    % Create empty EV string
    contrastString = '';
    % Run through all subjects and set their group
    for currEV = 1:M
        % Write to show images
        currString = ['# Display images for contrast_real ' num2str(currEV) '\n'];
        currString = [currString 'set fmri(conpic_real.' num2str(currEV) ') 1' '\n' '\n'];
        % Write title
        currString = [currString '# Title for contrast_real ' num2str(currEV) '\n'];
        currString = [currString 'set fmri(conname_real.' num2str(currEV) ') "' EVnames{currEV} '"' '\n' '\n'];        % Write contrast vector 
        for currElement = 1:M
            currString = [currString '# Real contrast_real vector ' num2str(currEV) ' element ' num2str(currElement) '\n'];
            % Only write a 1 for the element corresponding to the current EV
            if currElement == currEV
                currString = [currString 'set fmri(con_real' num2str(currEV) '.' num2str(currElement) ') 1' '\n' '\n'];
            else
                currString = [currString 'set fmri(con_real' num2str(currEV) '.' num2str(currElement) ') 0' '\n' '\n'];
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