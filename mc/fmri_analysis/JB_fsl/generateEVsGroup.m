function EVs = generateEVsGroup(subjectDir,subjectList,doPlot)
    % Extract all subjects from subjectList, containing tags separated by spaces
    subjectTags = strsplit(subjectList);
    % Get total number of subjects
    N = length(subjectTags);
    % Get subject files from filenames
    subjectFiles = cell(N,1);
    for currSubject = 1:N
        subjectFiles{currSubject} = [subjectTags{currSubject} '.mat'];
    end
    % Load them all
    subjects = cell(N,1);
    for currSubject = 1:N
        subjects{currSubject} = load(fullfile(subjectDir,subjectFiles{currSubject}));
        subjects{currSubject} = subjects{currSubject}.outputStruct;
    end
    
    % Read excel table into matrix
    subjectTable = readtable(fullfile(subjectDir,'ParticipantNotes.xlsx'),'Sheet','Subjects');
    % Find for each participant which row they correspond to
    subjectRow = zeros(N,1);
    for currSubject = 1:N
        subjectRow(currSubject) = find(contains(subjectTable.SubjectTag,subjectTags{currSubject}));
    end
    
    % Create EVs struct
    EVs = struct();
    
    % Start with the actual EV of interest: the mean
    EVs.Mean = ones(N,1);
    
    % Prediction performance on test blocks on day 3
    EVs.TaskTrials = zeros(N,1);
    for currSubject = 1:N
        EVs.TaskTrials(currSubject) = mean(max(0,subjects{currSubject}.Behaviour.Results.Session3.Trials.PredictionCorrect(ismember(subjects{currSubject}.Behaviour.Task.Session3.Trials.Block, [2 4])))); 
    end
    % Object question performance on day 3
    EVs.TaskObjectQuestions = zeros(N,1);
    for currSubject = 1:N
        EVs.TaskObjectQuestions(currSubject) = mean(max(0,subjects{currSubject}.Behaviour.Results.Session3.Questions.ObjectCorrect)); 
    end
    % Relation question performance on day 3 - this is very correlated to the object question, so leave it out
    %EVs.TaskRelationQuestions = zeros(N,1);
    %for currSubject = 1:N
    %    EVs.TaskRelationQuestions(currSubject) = mean(max(0,subjects{currSubject}.Behaviour.Results.Session3.Questions.RelationCorrect)); 
    %end 
    % Stories performance on day 3
    EVs.TaskStories = zeros(N,1);
    for currSubject = 1:N
        EVs.TaskStories(currSubject) = mean(max(0,subjects{currSubject}.Behaviour.Results.Session3.Stories.RelationCorrect)); 
    end      
    % Prediction performance in scan
    EVs.ScanTrials = zeros(N,1);
    for currSubject = 1:N
        EVs.ScanTrials(currSubject) = mean(subjects{currSubject}.Scan.Results.ScanTrials.Answers(subjects{currSubject}.Scan.Task.ScanTrials.Catch==1) == subjects{currSubject}.Scan.Task.ScanTrials.Relation(subjects{currSubject}.Scan.Task.ScanTrials.Catch==1)); 
    end     
    % Story performance in scan
    EVs.ScanStories = zeros(N,1);
    for currSubject = 1:N
        EVs.ScanStories(currSubject) = mean(subjects{currSubject}.Scan.Results.ScanStories.Answers(subjects{currSubject}.Scan.Task.ScanStories.Catch==1) == subjects{currSubject}.Scan.Task.ScanStories.Correct(subjects{currSubject}.Scan.Task.ScanStories.Catch==1)); 
    end 
    % Timeouts in scan
    EVs.TimeOuts = zeros(N,1);
    for currSubject = 1:N
        EVs.TimeOuts(currSubject) = sum((subjects{currSubject}.Scan.Results.ScanTrials.Responses(subjects{currSubject}.Scan.Task.ScanTrials.Catch==1) == 0)) + sum((subjects{currSubject}.Scan.Results.ScanStories.Responses(subjects{currSubject}.Scan.Task.ScanStories.Catch==1) == - 1) + (subjects{currSubject}.Scan.Task.ScanStories.Relation(subjects{currSubject}.Scan.Task.ScanStories.Catch==1) == 0)); 
    end  
    % Block order in scan
    EVs.BlockOrder = zeros(N,1);
    for currSubject = 1:N
        EVs.BlockOrder(currSubject) = subjects{currSubject}.Scan.Task.ScanBlocks(1); 
    end 
    
%     % Age
%     EVs.Age = zeros(N,1);
%     for currSubject = 1:N
%         EVs.Age(currSubject) = subjectTable.Age(subjectRow(currSubject));
%     end
%     % Gender
%     EVs.Gender = zeros(N,1);
%     for currSubject = 1:N
%         EVs.Gender(currSubject) = isequal(subjectTable.Gender(subjectRow(currSubject)),{'F'});
%     end
%     % Handedness
%     EVs.Hand = zeros(N,1);
%     for currSubject = 1:N
%         EVs.Hand(currSubject) = isequal(subjectTable.Hand(subjectRow(currSubject)),{'R'});
%     end    
    
    % Now reorganise EVs so they will be easy to compare later
    names = fieldnames(EVs);
    % Sort for standardisation
    names = sort(names);
    % Remove mean from EV names
    names = names(~ismember(names,'Mean'));
    % And add it again in front, so mean is always first EV
    names = ['Mean'; names];
    
    % Now collect all values in a new EV struct
    newEVs = struct();
    for currEV = 1:length(names)
        newEVs.(names{currEV}) = EVs.(names{currEV});
        % And demean if this is not the mean
        if ~isequal(names{currEV},'Mean')
            newEVs.(names{currEV}) = newEVs.(names{currEV}) - mean(newEVs.(names{currEV}));
        end
    end
    % And overwrite EVs with this new EVs struct
    EVs = newEVs;
    
    % Plot them all
    if doPlot
        disp(subjectTags);
        figure();
        for currEV = 1:length(names)
            subplot(3,4,currEV);
            hold on;
            plot([1,N],[0 0],'k--');
            plot(1:N,EVs.(names{currEV}));
            hold off;
            xlim([0, N+1]);
            title(names{currEV});
        end

        % Calculate corrcoefs
        allEVs = zeros(N,length(names));
        for currEV = 1:length(names)
            allEVs(:,currEV) = EVs.(names{currEV});
        end
        corrEVs = corrcoef(allEVs);
        % And plot correlation between them
        figure()
        corrImg = imagesc(corrEVs);
        set(corrImg,'XTick',1:length(names));
        set(corrImg,'XTickLabel',names);
        set(corrImg,'XTickLabelRotation',90);
        set(corrImg,'YTick',1:length(names));
        set(corrImg,'YTickLabel',names);        
        colorbar;
    end
end