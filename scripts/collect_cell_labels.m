% script to check whether electrode labels, region labels, and rois are coherent
% label-only QC, fast, simple, and easy to read
%%
clear all

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans";
if ~exist(source_dir, 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys';
end

deriv_dir = sprintf("%s/derivatives/", source_dir);

% load the label structure created by the label-collection script
if ~exist('label_info', 'var')
    load(fullfile(deriv_dir, 'electrode_label_info.mat'), 'label_info');
end

% ----------------------------
% PARAMETERS
% ----------------------------
n_cells = length(label_info.all_cells);

% output containers
qc_rows = struct([]);
flagged_rows = struct([]);

% counters
n_coherent = 0;
n_flagged = 0;
n_uninformative_electrode = 0;

% ----------------------------
% LOOP THROUGH ALL CELLS
% ----------------------------
for i = 1:n_cells

    curr_subject = label_info.all_cells(i).subject;
    curr_cell    = label_info.all_cells(i).cell_idx;

    curr_electrode = string(label_info.all_cells(i).electrodeLabel);
    curr_region    = string(label_info.all_cells(i).regionLabel);
    curr_roi       = string(label_info.all_cells(i).roi);
    curr_resolved  = string(label_info.all_cells(i).resolvedLabel);

    % parse each label into a broader anatomical family
    [electrode_family, electrode_side] = parse_label_family(curr_electrode);
    [region_family,    region_side]    = parse_label_family(curr_region);
    [roi_family,       roi_side]       = parse_label_family(curr_roi);

    % coherence logic:
    % - empty fields are allowed
    % - if at least two non-empty families agree, count as coherent
    % - if non-empty families conflict, flag it
    families = [electrode_family, region_family, roi_family];
    nonempty_families = families(families ~= "");

    if isempty(electrode_family)
        n_uninformative_electrode = n_uninformative_electrode + 1;
    end

    if isempty(nonempty_families)
        coherent = true;
        note = "no anatomical information in any field";
    else
        unique_families = unique(nonempty_families);

        if numel(unique_families) == 1
            coherent = true;
            note = "coherent";
        else
            coherent = false;
            note = "family mismatch";
        end
    end

    % side check only when sides are actually present
    side_fields = [electrode_side, region_side, roi_side];
    nonempty_sides = side_fields(side_fields ~= "");
    if ~isempty(nonempty_sides)
        unique_sides = unique(nonempty_sides);
        if numel(unique_sides) > 1
            coherent = false;
            if note == "coherent"
                note = "side mismatch";
            else
                note = note + "; side mismatch";
            end
        end
    end

    % make a tidy row
    qc_rows(i).subject          = curr_subject;
    qc_rows(i).cell_idx         = curr_cell;
    qc_rows(i).electrodeLabel   = curr_electrode;
    qc_rows(i).regionLabel      = curr_region;
    qc_rows(i).roi              = curr_roi;
    qc_rows(i).resolvedLabel    = curr_resolved;
    qc_rows(i).electrodeFamily   = electrode_family;
    qc_rows(i).regionFamily      = region_family;
    qc_rows(i).roiFamily         = roi_family;
    qc_rows(i).electrodeSide     = electrode_side;
    qc_rows(i).regionSide        = region_side;
    qc_rows(i).roiSide           = roi_side;
    qc_rows(i).coherent          = coherent;
    qc_rows(i).note              = note;

    if coherent
        n_coherent = n_coherent + 1;
    else
        n_flagged = n_flagged + 1;
        flagged_rows = [flagged_rows; qc_rows(i)]; %#ok<AGROW>
    end
end

% convert to table for easy viewing / saving
qc_table = struct2table(qc_rows);
flagged_table = struct2table(flagged_rows);

% summary
summary_table = table();
summary_table.n_cells = n_cells;
summary_table.n_coherent = n_coherent;
summary_table.n_flagged = n_flagged;
summary_table.n_uninformative_electrode = n_uninformative_electrode;

% show quick summary
disp(summary_table)
disp('--- flagged examples ---')
disp(flagged_table(1:min(20,height(flagged_table)), :))

% save outputs
save(fullfile(deriv_dir, 'label_qc.mat'), 'qc_table', 'flagged_table', 'summary_table');
writetable(qc_table, fullfile(deriv_dir, 'label_qc_table.csv'));
writetable(flagged_table, fullfile(deriv_dir, 'label_qc_flagged.csv'));

disp('DONE!')

%% ---------------------------------------------------------
function [family, side] = parse_label_family(label)
% Parse a label into a broad anatomical family and side.
% This is heuristic and meant for QC, not perfect annotation.

label = upper(strtrim(string(label)));
family = "";
side = "";

if label == ""
    return
end

% channel-like labels are usually not anatomically informative
if startsWith(label, "CHAN")
    return
end

% side detection for the common patterns in your labels
% examples:
%   LOFC      -> L
%   RpgACC    -> R
%   mLF2aCa01 -> L
%   mRT2bHb01 -> R
if startsWith(label, "ML") || startsWith(label, "MR")
    side = extractBetween(label, 2, 2);
elseif startsWith(label, "L") || startsWith(label, "R")
    side = extractBetween(label, 1, 1);
end

% broad family detection
if contains(label, "OFC")
    family = "OFC";
elseif contains(label, "ACC")
    family = "ACC";
elseif contains(label, "PCC")
    family = "PCC";
elseif contains(label, "EC")
    family = "EC";
elseif contains(label, "HC") || contains(label, "HIP")
    family = "HC";
elseif contains(label, "VCC")
    family = "VCC";
elseif contains(label, "MCC")
    family = "MCC";
elseif contains(label, "CC")
    family = "CC";
end
end