% extracts the channel labels for all utah datasets
% opens Electrodes.mat
% then takes ElecMapRaw and stores the first and seconds column:
% first column = anat locations
% seconds column = corresponding channel number

clear all

base_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
if ~exist(base_dir, 'dir')
    base_dir = '/ceph/behrens/svenja/human_ABCD_ephys'
end

deriv_dir = fullfile(base_dir, 'derivatives');

% define the utah subject list
session_list = [1,2,4,6,17,23,24,29,30,39,41,42,47,48, 52, 53, 54, 55];

% Loop through sessions
for i = 1:length(session_list)
    sesh_num = session_list(i);
    sesh_str = sprintf('s%02d', sesh_num);  % e.g., s01, s02, ...
    
    % Build full path to the .mat file
    elec_path = fullfile(base_dir, sesh_str, 'electrodes', 'Electrodes.mat');
    deriv_path = fullfile(base_dir, 'derivatives', sesh_str, 'LFP');
    % Create output directory if it doesn't exist
    if ~exist(deriv_path, 'dir')
        mkdir(deriv_path);
    end
    % Check if file exists
    if isfile(elec_path)
        fprintf("Processing %s...\n", elec_path);
        
        % Load the .mat file
        load(elec_path)

        % Extract the column of channel numbers
        channel_numbers = cell2mat(ElecMapRaw(:,2));
        
        % Get sorting indices
        [~, sort_idx] = sort(channel_numbers);
        
        % Apply sorting to all rows of ElecMapRaw
        ElecMapSorted = ElecMapRaw(sort_idx, :);
        
        writecell(ElecMapSorted(:, 1:2), sprintf("%s/utah_elec_labels_%02d.csv", deriv_path, sesh_num));

        disp('now storing:')
        disp(sprintf("%s/utah_elec_labels_%02d.csv", deriv_path, sesh_num))
    else 
        disp(sprintf("no electrode label file for session %02d", sesh_num))
    end
end

