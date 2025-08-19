% load your one-file QC result (so we can include only passed cells)

% ----------- CONFIG -----------
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans";
if ~exist(char(source_dir), 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys';
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_24-Apr-2025.mat", source_dir));
else
    abcd_data = load(sprintf("%s/abcd_data_10-Jul-2025.mat", source_dir));
end

deriv_dir = fullfile(char(source_dir), 'derivatives');
if ~exist(deriv_dir,'dir'); mkdir(deriv_dir); end

% Load the one-file QC results so we can use pass masks
S = load(fullfile(deriv_dir, 'qc_all_sessions.mat'), 'qc_all');

for sub = 1:length(abcd_data.abcd_data)
    subj = abcd_data.abcd_data(sub);

    %pass_mask = [];
    sess_id = sprintf('s%02d', sub);
    if isfield(S, 'qc_all') && sub <= numel(S.qc_all.sessions)
        %pass_mask = S.qc_all.sessions(sub).pass_mask;
        sess_id   = S.qc_all.sessions(sub).subject_id;
    end

    % Quick & dirty: task window, 50 ms bins, passed cells only
    plot_session_correlogram(subj, ...
        'SessionID', sess_id, 'SessionIndex', sub, ...
        'UsePassedOnly', false, ...
        'Window', 'task', 'BinSizeS', 0.05);

    % If you prefer only the overlap across all included cells:
    % plot_session_correlogram(subj, 'SessionID', sess_id, 'SessionIndex', sub, ...
    %     'PassMask', pass_mask, 'UsePassedOnly', true, 'Window', 'overlap', 'BinSizeS', 0.05);
end
