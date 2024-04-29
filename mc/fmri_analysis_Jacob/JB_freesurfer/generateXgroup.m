function generateXgroup(subjectDir, subjectList, outDir)
    % Generate a design matrix file for group analysis in freesurfer
    
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
    % Output state
    disp('Finished generating group EVs with names:');
    disp(EVnames);
    
    % Create design matrix
    X = zeros(N,M);
    % Fill columns with EVs
    for currEV = 1:M
        X(:,currEV) = EVs.(EVnames{currEV});
    end
    % Output state
    disp('Finished writing design matrix:');
    disp(X);
    
    % Store design matrix as mat file    
    save(fullfile(outDir,'X.mat'),'X','-v4');
    disp(['Finished storing file as ' fullfile(outDir,'X.mat')]);
    
    % Create contrast: one for first EV (mean), zero for all others
    C = zeros(1,M); C(1) = 1;
    
    % Convert the contrast into a ASCII string
    outputString = mat2str(C);
    outputString = outputString(2:(end-1));
    outputString = [outputString '\n'];
    % Write output to new file
    fid = fopen(fullfile(outDir,'C.txt'),'wt');
    fprintf(fid, outputString); 
    fclose(fid);        
end