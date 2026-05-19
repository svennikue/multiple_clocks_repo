% script to collect electrode labels, region labels, and rois only
% fast, simple, and label-focused
%%

clear all

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans";
if ~exist(source_dir, 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys';
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_08-Sep-2025.mat", source_dir));
else
    abcd_data = load(sprintf("%s/derivatives/abcd_passed.mat", source_dir));
end

deriv_dir = sprintf("%s/derivatives/", source_dir);

% subject list
subject_list = 1:length(abcd_data.abcd_passed.abcd_data);

% output structure
label_info = struct();
label_info.subject = struct([]);
label_info.all_cells = struct([]);
label_info.unique_electrode_labels = {};
label_info.unique_region_labels    = {};
label_info.unique_rois             = {};

% optional: keep everything in one growing list first
all_electrode_labels = {};
all_region_labels    = {};
all_rois             = {};
all_resolved_labels  = {};

cell_counter = 0;

% LOOP THROUGH SUBJECTS
for sub = 1:length(subject_list)
    disp(sprintf('...now processing subject %d', sub));

    subj = abcd_data.abcd_passed.abcd_data(sub);
    subject_label = abcd_data.abcd_passed.abcd_data(sub).subject_ID;

    % per-subject containers
    subj_electrode_labels = {};
    subj_region_labels = {};
    subj_rois = {};
    subj_resolved_labels = {};

    % LOOP THROUGH CELLS
    for cell_idx = 1:length(subj.neural_data)

        % raw electrode label
        if isfield(subj.neural_data(cell_idx), 'electrodeLabel') && ...
                ~isempty(subj.neural_data(cell_idx).electrodeLabel)
            curr_electrode_label = string(subj.neural_data(cell_idx).electrodeLabel);
        else
            curr_electrode_label = "";
        end

        % region label
        if isfield(subj.neural_data(cell_idx), 'regionLabel') && ...
                ~isempty(subj.neural_data(cell_idx).regionLabel)
            curr_region_label = string(subj.neural_data(cell_idx).regionLabel);
        else
            curr_region_label = "";
        end

        % roi label
        curr_roi = "";
        if isfield(subj.neural_data(cell_idx), 'roi') && ~isempty(subj.neural_data(cell_idx).roi)
            if ~isempty(subj.neural_data(cell_idx).roi{1})
                curr_roi = string(subj.neural_data(cell_idx).roi{1});
            end
        end

        % label used in the original script when roi is missing
        if curr_roi ~= ""
            curr_resolved_label = curr_roi;
        else
            curr_resolved_label = curr_region_label;
        end

        % store per-cell entry
        cell_counter = cell_counter + 1;
        label_info.all_cells(cell_counter).subject = sub;
        label_info.all_cells(cell_counter).subject_label = subject_label;
        label_info.all_cells(cell_counter).cell_idx = cell_idx;
        label_info.all_cells(cell_counter).electrodeLabel = curr_electrode_label;
        label_info.all_cells(cell_counter).regionLabel = curr_region_label;
        label_info.all_cells(cell_counter).roi = curr_roi;
        label_info.all_cells(cell_counter).resolvedLabel = curr_resolved_label;

        % store per-subject lists
        subj_electrode_labels{end+1,1} = char(curr_electrode_label);
        subj_region_labels{end+1,1}    = char(curr_region_label);
        subj_rois{end+1,1}             = char(curr_roi);
        subj_resolved_labels{end+1,1}   = char(curr_resolved_label);

        % store global lists
        all_electrode_labels{end+1,1} = char(curr_electrode_label);
        all_region_labels{end+1,1}    = char(curr_region_label);
        all_rois{end+1,1}             = char(curr_roi);
        all_resolved_labels{end+1,1}  = char(curr_resolved_label);

    end

    % save subject-level label structure
    label_info.subject(sub).subject_id = sub;
    label_info.subject(sub).electrode_labels = subj_electrode_labels;
    label_info.subject(sub).region_labels    = subj_region_labels;
    label_info.subject(sub).rois             = subj_rois;
    label_info.subject(sub).resolved_labels  = subj_resolved_labels;
end

% unique labels across all subjects
label_info.unique_electrode_labels = unique(all_electrode_labels);
label_info.unique_region_labels    = unique(all_region_labels);
label_info.unique_rois             = unique(all_rois);
label_info.unique_resolved_labels  = unique(all_resolved_labels);

% save
save(fullfile(deriv_dir, 'electrode_label_info-2026-may.mat'), 'label_info');

disp('DONE!');