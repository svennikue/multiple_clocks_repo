function RSAGenerateEVs(inputFile, inputFolder, useAllRels, separateBlocks, doPlot, timeWindow)
    % Load data from subject data object and save to output struct for use in SPM
    % If useAllRels is false, keep position vs story prediction blocks separate
    % If useAllRels is true, make EVs for relations across position vs story prediction blocks
    % Concretely, the regressors generated will be as follows:
    % [] only exist if useAllRels = false, {} only exist if separateBlocks = true
    % 1. [Stim {block 1}]
    % 2. [Stim {block 2}]
    % 3. Rel {block 1}
    % 4. Rel {block 2}
    % TimeWindow is optional: it can be set to include regressors with activity within
    % a specific timewindow, to allow for running separate GLMs. 
    % Set to [0, inf] by default so all regressors are included
    if nargin < 6
        timeWindow = [0, inf];
    end
    
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
    
    % Number of stimuli AND number of relations - equal for me so I can write neater code
    N = max(subject.Scan.Task.ScanTrials.Stimulus);

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
        
    % We now want catch trials to be treated identical to but separately from normal trials, so repeat with small differences
    for isCatch = [false, true]
        % Offset to start filling out EVs for catch trials, after normal trials
        catchOffset = isCatch * N * (2 + 2 * separateBlocks);
        % Name to add to EV name for catch trials
        if isCatch
            catchName = 'Catch';
        else
            catchName = '';        
        end
        % Run through relations or stimuli, generating EVs for each
        for currStimRel = 1:N
            % Use currStimRel as index for scan trials
            trialsEV1 = currStimRel + catchOffset;
            % If you want to separate blocks 1 and 2 for trials and stories: use N+currStimRel for second trial block
            if separateBlocks
                trialsEV2 = currStimRel + N + catchOffset;
            else
                trialsEV2 = [];
            end
            % If you want to separate blocks 1 and 2 for trials and stories: use N*2+currStimRel for first story block
            if separateBlocks
                storiesEV1 = currStimRel + N*2 + catchOffset;
            else
                storiesEV1 = currStimRel + N + catchOffset;
            end        
            % If you want to separate blocks 1 and 2 for trials and stories: use N*3+currStimRel for second story block
            if separateBlocks
                storiesEV2 = currStimRel + N*3 + catchOffset;
            else
                storiesEV2 = [];
            end        
            % Scan trials block 1
            if separateBlocks
                names{trialsEV1} = ['Stim' catchName num2str(currStimRel) 'Block1'];
            else
                names{trialsEV1} = ['Stim' catchName num2str(currStimRel)];
            end
            blocks{trialsEV1} = 'ScanTrials';
            values{trialsEV1} = double(subject.Scan.Task.ScanTrials.Stimulus == currStimRel);       
            % Scan trials block 2
            if separateBlocks
                names{trialsEV2} = ['Stim' catchName num2str(currStimRel) 'Block2'];
                blocks{trialsEV2} = 'ScanTrials';
                values{trialsEV2} = double(subject.Scan.Task.ScanTrials.Stimulus == currStimRel);        
            end
            % Scan stories block 1
            if separateBlocks
                names{storiesEV1} = ['Rel' catchName num2str(currStimRel) 'Block1'];
            else
                names{storiesEV1} = ['Rel' catchName num2str(currStimRel)];
            end
            blocks{storiesEV1} = 'ScanStories';
            values{storiesEV1} = double(subject.Scan.Task.ScanStories.Relation == currStimRel);          
            % Scan stories block 2
            if separateBlocks
                names{storiesEV2} = ['Rel' catchName num2str(currStimRel) 'Block2'];
                blocks{storiesEV2} = 'ScanStories';
                values{storiesEV2} = double(subject.Scan.Task.ScanStories.Relation == currStimRel);            
            end
            % Then: get keep, start, and duration for both (since calculating them is identical)
            for currEV = [trialsEV1 trialsEV2 storiesEV1 storiesEV2]
                % Calculate start and duration for all stimuli/relations - we'll throw away all unnecessary ones later
                if calculateDuration
                    % Read start time from recorded behaviour during scanning
                    start{currEV} = subject.Scan.Results.(blocks{currEV}).StartTimes - triggerTime;
                    % Bit ugly: read duration from difference between start times; last trial simply minimum of differences (ok for catch because they will be removed)
                    duration{currEV} = [diff(subject.Scan.Results.(blocks{currEV}).StartTimes); min(diff(subject.Scan.Results.(blocks{currEV}).StartTimes))];
                    % Also fix duration on trials before block endings
                    duration{currEV}(diff(subject.Scan.Task.(blocks{currEV}).Block)>0) = min(diff(subject.Scan.Results.(blocks{currEV}).StartTimes));
                else
                    % Read start time from recorded behaviour during scanning
                    start{currEV} = subject.Scan.Results.(blocks{currEV}).StimulusStart - triggerTime;
                    % And read duration from recorded behaviour during scanning
                    duration{currEV} = subject.Scan.Results.(blocks{currEV}).StimulusDuration;
                end        
                % Find catch trials
                catchTrials = subject.Scan.Task.(blocks{currEV}).Catch;
                % Only keep trials that are A) not catch trials for normal run, or catch trials for catch run and B) have value 1
                keep{currEV} = ((~isCatch & ~catchTrials) | (isCatch & catchTrials)) & boolean(values{currEV});
                % And if you are separating blocks: also only keep trials in current block
                if separateBlocks
                    % Get block for current EV
                    currBlock = mod(find(currEV==[trialsEV1, trialsEV2 storiesEV1 storiesEV2])-1,2)+1;
                    % And only keep trials in that block
                    keep{currEV} = keep{currEV} & subject.Scan.Task.(blocks{currEV}).Block == currBlock;
                end
            end
        end
    end
    
    % Set analysis fields for looping over trials and stories
    analysisFields = {'ScanTrials','ScanStories'};
    analysisNames = {'Stim', 'Rel'}; 
    
    % Add motor response nuisance EV, making sure it can be split between blocks
    for currAnalysisField = 1:length(analysisFields)
        % If you are separating blocks: add suffix so you run through both blocks
        if separateBlocks
            suffix = {'Block1','Block2'};
        else
            % Else: no block suffix
            suffix = {''};
        end
        % Run through blocks
        for currBlock = 1:length(suffix)
            names{end+1} = [analysisNames{currAnalysisField} 'Response' suffix{currBlock}];
            blocks{end+1} = analysisFields{currAnalysisField};
            values{end+1} = subject.Scan.Task.(blocks{end}).Catch;
            if calculateDuration
                start{end+1} = subject.Scan.Results.(blocks{end}).StartTimes + min(diff(subject.Scan.Results.(blocks{end}).StartTimes)) - triggerTime;
                duration{end+1} = subject.Scan.Results.(blocks{end}).ResponseTimes;
            else
                start{end+1} = subject.Scan.Results.(blocks{end}).ResponseStart - triggerTime;
                duration{end+1} = subject.Scan.Results.(blocks{end}).ResponseDuration;
            end        
            % Let's include all responses here, whether a button was pressed or not - but only from right block
            keep{end+1} = logical(values{end}) & (~separateBlocks | subject.Scan.Task.(blocks{currEV}).Block == currBlock);
        end
    end       
    
    % If you want to use relations instead of stims: merge EVs between stories and trials 
    if useAllRels
        % If you are separating blocks: add suffix so you run through both blocks
        if separateBlocks
            suffix = {'Block1','Block2'};
        else
            % Else: no block suffix
            suffix = {''};
        end      
        % Run through both blocks in case of separating blocks; else, just the combined blocks
        for currSuffix = suffix
            % Combine response
            currRelEntry = find(contains(names, ['RelResponse' currSuffix{1}]));
            currStimEntry = find(contains(names,['StimResponse' currSuffix{1}]));
            % Merge the two, adding stim entry to rel entry then removing stim entry
            [names, blocks, values, start, duration, keep] = ...
                addJtoI(currRelEntry, currStimEntry, names,blocks,values,start,duration,keep);            
            % Combine catch trials and normal trials
            for isCatch = [false, true]
                % Name to add to EV name for catch trials
                if isCatch
                    catchName = 'Catch';
                else
                    catchName = '';        
                end
                % Then run through relations
                for currRel = 1:N
                    % Find which entry hold this relation
                    currRelEntry = find(contains(names, ['Rel' catchName num2str(currRel) currSuffix{1}]));
                    % Find which stimulus has the current relation
                    currStim = find(subject.Subject.RelationMap==currRel);
                    % Find cell entry entry of that stim
                    currStimEntry = find(contains(names,['Stim' catchName num2str(currStim) currSuffix{1}]));
                    % Merge the two, adding stim entry to rel entry then removing stim entry
                    [names, blocks, values, start, duration, keep] = ...
                        addJtoI(currRelEntry, currStimEntry, names,blocks,values,start,duration,keep);                                  
%                     % Concatenate values, keep, start, and duration with story trials with the same relations
%                     values{currRelEntry} = [values{currStimEntry}; values{currRelEntry}];
%                     keep{currRelEntry} = [keep{currStimEntry}; keep{currRelEntry}];
%                     start{currRelEntry} = [start{currStimEntry}; start{currRelEntry}];
%                     duration{currRelEntry} = [duration{currStimEntry}; duration{currRelEntry}];
%                     % And remove stim entry from all cell arrays
%                     names = names([1:(currStimEntry-1) (currStimEntry+1):end]);
%                     blocks = blocks([1:(currStimEntry-1) (currStimEntry+1):end]);
%                     values = values([1:(currStimEntry-1) (currStimEntry+1):end]);
%                     keep = keep([1:(currStimEntry-1) (currStimEntry+1):end]);
%                     start = start([1:(currStimEntry-1) (currStimEntry+1):end]);
%                     duration = duration([1:(currStimEntry-1) (currStimEntry+1):end]);                
                end
            end
        end
    end
        
    % Clean all EV values and sort by start time
    for currEV = 1:length(names)
        % Select only values where keep is true
        values{currEV} = values{currEV}(keep{currEV});
        start{currEV} = start{currEV}(keep{currEV});
        duration{currEV} = duration{currEV}(keep{currEV}); 
        % Finally: sort all by start, because it's nicer to plot
        [~, sortedIds] = sort(start{currEV});
        values{currEV} = values{currEV}(sortedIds);
        start{currEV} = start{currEV}(sortedIds);
        duration{currEV} = duration{currEV}(sortedIds);
    end
    
    % Now take care of time window. For each EV, check if it's active during the time window
    inWindow = true(size(names));    
    % If it is, keep its activation within the window, and update the start values to new start time
    for currEV = 1:length(names)
        activeInWindow = (timeWindow(1) < start{currEV} & start{currEV} < timeWindow(2)) | ...
            (timeWindow(1) < (start{currEV} + duration{currEV}) & (start{currEV} + duration{currEV}) < timeWindow(2));
        if any(activeInWindow)
            start{currEV} = start{currEV}(activeInWindow) - timeWindow(1);
            values{currEV} = values{currEV}(activeInWindow);
            duration{currEV} = duration{currEV}(activeInWindow);
        else
            inWindow(currEV) = false;
        end
    end
    % If it is not, remove the EV
    names = names(inWindow);
    values = values(inWindow);    
    start = start(inWindow);
    duration = duration(inWindow);
    
    % Set orthogonalisation for all EVs to false
    orthogonalise = num2cell(false(size(names)));
    
    % Create struct with names that SPM expects
    outputStruct = struct();
    outputStruct.names = names;
    outputStruct.onsets = start;
    outputStruct.durations = duration;
    outputStruct.orth = orthogonalise;
    
    % Finally: store all these EVs in single .mat file for later use with SPM
    save(fullfile(inputFolder,'EVs.mat'),'-struct','outputStruct');
    
    % Return EVs for plotting
    EVs = struct();
    for currEV = 1:length(names)
        % Create plottable x-axis
        EVs.(names{currEV}).t = [start{currEV}-0.001 start{currEV} start{currEV}+duration{currEV} start{currEV}+duration{currEV}+0.001]';
        EVs.(names{currEV}).t = EVs.(names{currEV}).t(:);
        % Create plottable y-axis
        EVs.(names{currEV}).y = [zeros(size(values{currEV})) values{currEV} values{currEV} zeros(size(values{currEV}))]';
        EVs.(names{currEV}).y = EVs.(names{currEV}).y(:);
        % And store the other values
        EVs.(names{currEV}).Values = values{currEV};
        EVs.(names{currEV}).Start = start{currEV};
        EVs.(names{currEV}).Duration = duration{currEV};
    end
    
    % And plot results
    if doPlot
        for currEV = 1:length(values)
            figure();
            plot(EVs.(names{currEV}).t, EVs.(names{currEV}).y);
            xlabel('Time (s)');
            ylabel('Amplitude (a.u.)');
            ylim([-0.1 1.1]);
            title(names{currEV});
        end
    end        
    
    function [names, blocks, values, start, duration, keep] = mergeLastTwo(names,blocks,values,start,duration,keep)
        % Remove last names and blocks
        names = names(1:(end-1));
        blocks{end-1} = 'Both'; blocks = blocks(1:(end-1));    
        % Concatenate values, start, duration, and keep, and remove last entry
        values{end-1} = [values{end-1}; values{end}]; values = values(1:(end-1));
        start{end-1} = [start{end-1}; start{end}]; start = start(1:(end-1));
        duration{end-1} = [duration{end-1}; duration{end}]; duration = duration(1:(end-1));
        keep{end-1} = [keep{end-1}; keep{end}]; keep = keep(1:(end-1));
    end

    function [names, blocks, values, start, duration, keep] = addJtoI(i, j, names,blocks,values,start,duration,keep)
        % Concatenate values, keep, start, and duration with story trials from j to i
        values{i} = [values{j}; values{i}];
        keep{i} = [keep{j}; keep{i}];
        start{i} = [start{j}; start{i}];
        duration{i} = [duration{j}; duration{i}];
        % And remove j from all cell arrays
        names = names([1:(j-1) (j+1):end]);
        blocks = blocks([1:(j-1) (j+1):end]);
        values = values([1:(j-1) (j+1):end]);
        keep = keep([1:(j-1) (j+1):end]);
        start = start([1:(j-1) (j+1):end]);
        duration = duration([1:(j-1) (j+1):end]);      
    end
end