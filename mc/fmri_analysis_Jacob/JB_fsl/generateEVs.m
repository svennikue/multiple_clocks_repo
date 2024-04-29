function generateEVs(inputFile, outputFolder, separateBlocks, doPlot)
    % Load data from subject data object and save to output folder
    
    % Load subject file
    subject = load(inputFile);
    subject = subject.outputStruct;
    
    % Check if this subject file contains durations, or that they need to be calculated - this would be neat, but cluster runs matlab 2016
    % calculateDuration = ~any(contains(fieldnames(subject.Scan.Results.ScanTrials),'StimulusDuration'));
    % So I'll have to write an ugly loop to find if StimulusDuration field exists
    calculateDuration = true;
    trialFields = fieldnames(subject.Scan.Results.ScanTrials);
    for currField = 1:length(trialFields)
        if ~isempty(strfind(trialFields{currField},'StimulusDuration'))
            calculateDuration = false;
        end
    end    
    
    % Read trigger time from scan results: trigger time defines 0 time in scan
    triggerTime = subject.Scan.Results.Time.TriggerTime;    
    
    % Keep track of explanatory variable names
    names = {};
    % Keep track of block type that EV occurs in
    blocks = {};
    % Store EV values
    values = {};
    % Keep track of when stimulus starts
    start = {};
    % Keep track of stimulus duration
    duration = {};
    % Remove values at trials that don't apply for certain EV
    keep = {};
    
    % Run through analysis suppression hypotheses
    analysisFields = {'ScanTrials','ScanStories'}; % Can't take all fields anymore since there is also a similarity field!
    for currAnalysisField = 1:length(analysisFields)
        hypothesisFields = fieldnames(subject.Analysis.(analysisFields{currAnalysisField}));
        for currHypothesisField = 1:length(hypothesisFields)
            % Copy all information from this suppression hypothesis into EV cell array
            names{end+1} = hypothesisFields{currHypothesisField};
            blocks{end+1} = analysisFields{currAnalysisField};
            values{end+1} = subject.Analysis.(blocks{end}).(names{end});
            % If you want to separate identical EVs in different blocks: add suffix
            if separateBlocks
                names{end} = [names{end} num2str(currAnalysisField)];
            end
            % Expected bold signal is inverted suppression
            values{end} = max(values{end}) - values{end};    
            if calculateDuration
                % Read start time from recorded behaviour during scanning
                start{end+1} = subject.Scan.Results.(blocks{end}).StartTimes - triggerTime;
                % Bit ugly: read duration from difference between start times; last trial simply minimum of differences (ok for catch because they will be removed)
                duration{end+1} = [diff(subject.Scan.Results.(blocks{end}).StartTimes); min(diff(subject.Scan.Results.(blocks{end}).StartTimes))];
                % Also fix duration on trials before block endings
                duration{end}(diff(subject.Scan.Task.(blocks{end}).Block)>0) = min(diff(subject.Scan.Results.(blocks{end}).StartTimes));
            else
                % Read start time from recorded behaviour during scanning
                start{end+1} = subject.Scan.Results.(blocks{end}).StimulusStart - triggerTime;
                % And read duration from recorded behaviour during scanning
                duration{end+1} = subject.Scan.Results.(blocks{end}).StimulusDuration;
            end
            % Find catch trials and trials that immediately follow
            catchTrials = ismember(1:length(values{end}),[find(subject.Scan.Task.(blocks{end}).Catch);find(subject.Scan.Task.(blocks{end}).Catch)+1]);
            % Only keep trials that are not catch trials or immediately follow those
            keep{end+1} = ~catchTrials;            
        end
    end
        
    % Add catch trial nuisance EV
    for currAnalysisField = 1:length(analysisFields)
        names{end+1} = 'Catch';
        blocks{end+1} = analysisFields{currAnalysisField};
        values{end+1} = subject.Scan.Task.(blocks{end}).(names{end});
        if calculateDuration
            start{end+1} = subject.Scan.Results.(blocks{end}).StartTimes - triggerTime;
            duration{end+1} = [diff(subject.Scan.Results.(blocks{end}).StartTimes); min(diff(subject.Scan.Results.(blocks{end}).StartTimes))];
            duration{end}(diff(subject.Scan.Task.(blocks{end}).Block)>0) = min(diff(subject.Scan.Results.(blocks{end}).StartTimes));        
        else
            start{end+1} = subject.Scan.Results.(blocks{end}).StimulusStart - triggerTime;
            duration{end+1} = subject.Scan.Results.(blocks{end}).StimulusDuration;
        end
        keep{end+1} = logical(values{end});
    end    
    
    % Add trial-after-catch trial nuisance EV
    for currAnalysisField = 1:length(analysisFields)
        names{end+1} = 'AfterCatch';
        blocks{end+1} = analysisFields{currAnalysisField};
        values{end+1} = circshift(subject.Scan.Task.(blocks{end}).Catch,1);
        values{end}(1) = 0;
        if calculateDuration
            start{end+1} = subject.Scan.Results.(blocks{end}).StartTimes - triggerTime;
            duration{end+1} = [diff(subject.Scan.Results.(blocks{end}).StartTimes); min(diff(subject.Scan.Results.(blocks{end}).StartTimes))];
            duration{end}(diff(subject.Scan.Task.(blocks{end}).Block)>0) = min(diff(subject.Scan.Results.(blocks{end}).StartTimes));                    
        else
            start{end+1} = subject.Scan.Results.(blocks{end}).StimulusStart - triggerTime;
            duration{end+1} = subject.Scan.Results.(blocks{end}).StimulusDuration;
        end
        keep{end+1} = logical(values{end});
    end      
    
    % Add trial onset nuisance EV
    for currAnalysisField = 1:length(analysisFields)
        names{end+1} = 'Onset';
        blocks{end+1} = analysisFields{currAnalysisField};
        values{end+1} = ones(size(subject.Scan.Task.(blocks{end}).Block));
        if calculateDuration
            start{end+1} = subject.Scan.Results.(blocks{end}).StartTimes - triggerTime;
        else
            start{end+1} = subject.Scan.Results.(blocks{end}).StimulusStart - triggerTime;
        end        
        duration{end+1} = 0.01 * ones(size(values{end}));
        keep{end+1} = logical(values{end});
    end         
    
    % Add motor response nuisance EV
    for currAnalysisField = 1:length(analysisFields)
        names{end+1} = 'Response';
        blocks{end+1} = analysisFields{currAnalysisField};
        values{end+1} = subject.Scan.Task.(blocks{end}).Catch;
        if calculateDuration
            start{end+1} = subject.Scan.Results.(blocks{end}).StartTimes + min(diff(subject.Scan.Results.(blocks{end}).StartTimes)) - triggerTime;
            duration{end+1} = subject.Scan.Results.(blocks{end}).ResponseTimes;
        else
            start{end+1} = subject.Scan.Results.(blocks{end}).ResponseStart - triggerTime;
            duration{end+1} = subject.Scan.Results.(blocks{end}).ResponseDuration;
        end        
        % To decide: should you model the time-out cases? Probably should model separately - but in general there are few of them
        keep{end+1} = logical(values{end} & subject.Scan.Results.(blocks{end}).Answers ~= 0);
    end       
    
    % Find all location where each name occurs
    concatNames = unique(names);
    % Concatenate all EVs with identical names: turn the same EVs in different blocks into one single EV
    concatValues = cell(1,length(concatNames));
    concatStart = cell(1,length(concatNames));
    concatDuration = cell(1,length(concatNames));    
    concatKeep = cell(1,length(concatNames));        
    for currEV = 1:length(concatNames)
        % Find where current name occurs - this would be the cleanest, but cluster runs R2016a
        % occurrences = find(contains(names,concatNames{currEV}));
        % So I'll have to write an ugly loop to find where the concat name occurs in the original names array
        occurrences = [];
        for currName = 1:length(names)
            if isequal(names{currName}, concatNames{currEV})
                occurrences = [occurrences currName];
            end
        end
        for currOccurence = occurrences
            % Concatenate values, start, duration, and keep between all occurrences of this name
            concatValues{currEV} = [concatValues{currEV}; values{currOccurence}];
            concatStart{currEV} = [concatStart{currEV}; start{currOccurence}];
            concatDuration{currEV} = [concatDuration{currEV}; duration{currOccurence}];
            concatKeep{currEV} = [logical(concatKeep{currEV}); keep{currOccurence}];            
        end
        % Finally: sort all by start, because it's nicer to plot
        [~, sortedIds] = sort(concatStart{currEV});
        % And rearrange all values to reflect sorted start time
        concatValues{currEV} = concatValues{currEV}(sortedIds);
        concatStart{currEV} = concatStart{currEV}(sortedIds);
        concatDuration{currEV} = concatDuration{currEV}(sortedIds);
        concatKeep{currEV} = concatKeep{currEV}(sortedIds);   
    end
        
    % Clean and demean all EV values
    for currEV = 1:length(concatValues)
        % Select only values where keep is true
        concatValues{currEV} = concatValues{currEV}(concatKeep{currEV});
        concatStart{currEV} = concatStart{currEV}(concatKeep{currEV});
        concatDuration{currEV} = concatDuration{currEV}(concatKeep{currEV}); 
        % For parametric regressors: demean (since the mean/general activation is caught by offset (or for me: onset) regressors)
        if length(unique(concatValues{currEV}))>1
            % Scale all values to be between 0 and 1 (not strictly necessary but nice for comparability)   
            concatValues{currEV} = (concatValues{currEV} - min(concatValues{currEV})) / (max(concatValues{currEV}) - min(concatValues{currEV}));
            % And demean
            concatValues{currEV} = concatValues{currEV} - mean(concatValues{currEV});
        end
    end
    
    % Finally: store all these EVs as tsv files
    for currEV = 1:length(concatValues)
        dlmwrite(fullfile(outputFolder,[concatNames{currEV} '.txt']),[concatStart{currEV} concatDuration{currEV} concatValues{currEV}],'delimiter','\t');
    end
    
    % Return EVs for plotting
    EVs = struct();
    for currEV = 1:length(concatValues)
        % Create plottable x-axis
        EVs.(concatNames{currEV}).t = [concatStart{currEV}-0.001 concatStart{currEV} concatStart{currEV}+concatDuration{currEV} concatStart{currEV}+concatDuration{currEV}+0.001]';
        EVs.(concatNames{currEV}).t = EVs.(concatNames{currEV}).t(:);
        % Create plottable y-axis
        EVs.(concatNames{currEV}).y = [zeros(size(concatValues{currEV})) concatValues{currEV} concatValues{currEV} zeros(size(concatValues{currEV}))]';
        EVs.(concatNames{currEV}).y = EVs.(concatNames{currEV}).y(:);
        % And store the other values
        EVs.(concatNames{currEV}).Values = concatValues{currEV};
        EVs.(concatNames{currEV}).Start = concatStart{currEV};
        EVs.(concatNames{currEV}).Duration = concatDuration{currEV};
    end
    
    % And plot results
    if doPlot
        for currEV = 1:length(concatValues)
            figure();
            plot(EVs.(concatNames{currEV}).t, EVs.(concatNames{currEV}).y);
            xlabel('Time (s)');
            ylabel('Amplitude (a.u.)');
            title(concatNames{currEV});
        end
    end        
end