function roi_tuning = characterize_tuning(model,clusters,labels,ims,int,infun)

%Load images
%dummy_image = imread(ims{1});
%is = size(dummy_image);
is = [100,150,3];
if ~isempty(ims),
image_mat = load_and_crop(ims,is);
else
    image_mat = [];
end

roi_tuning = cell(numel(model),1);
for idx = 1:numel(model), %iterate through masks,
    tm = model{idx};
    tc = clusters{idx};
    if int == true,
        tm = tm(2:end,:);
    end
    %%%%Use abs?
    %tm = abs(tm);
    %%%%
    uni_clusts = unique(tc);
    num_clusts = numel(uni_clusts);
    mask_tuning = zeros(size(tm,1),num_clusts);
    for il = 1:num_clusts,
        figure,
        it_data = tm(:,tc == uni_clusts(il));
        it_vals = infun(max(it_data,[],2));
        mask_tuning(:,il) = it_vals;
        [sorted_vals,sorted_ids] = sort(it_vals,'descend');
        subplot(2,1,1),
        stem(sorted_vals),
        set(gca,'XTick',1:numel(labels),'XTickLabels',labels(sorted_ids))
        xticklabel_rotate([],45,[],'Fontsize',8);
        if ~isempty(image_mat),
            scaled_vals = 1 - (1/numel(labels) - (exp(-it_vals) ./ sum(exp(-it_vals)))) * 100;
            colors = bsxfun(@times,summer(numel(labels)),255);
            for pi = 1:numel(labels),
                scaled_im = image_mat(:,:,:,pi);
                fi = find(sorted_ids==pi);
                scaled_im = addborder(scaled_im,numel(labels) + 1 - fi,colors(fi,:),'outer');
                %scaled_im = imresize(scaled_im,scaled_vals(pi));
                subplot(8,5,pi + numel(labels));
                imshow(mat2gray(scaled_im));
                title(sprintf('%s',labels{pi}));
            end
        end
    end
    roi_tuning{idx} = mask_tuning;
end


function images = load_and_crop(ims,is)

images = zeros(is(1),is(2),is(3),numel(ims));
for idx = 1:numel(ims),
    ti = imresize(imread(ims{idx}),[is(1),is(2)]);
    images(:,:,:,idx) = ti;
end

function hText = xticklabel_rotate(XTick,rot,varargin)
%hText = xticklabel_rotate(XTick,rot,XTickLabel,varargin)     Rotate XTickLabel
%
% Syntax: xticklabel_rotate
%
% Input:
% {opt}     XTick       - vector array of XTick positions & values (numeric)
%                           uses current XTick values or XTickLabel cell array by
%                           default (if empty)
% {opt}     rot         - angle of rotation in degrees, 90° by default
% {opt}     XTickLabel  - cell array of label strings
% {opt}     [var]       - "Property-value" pairs passed to text generator
%                           ex: 'interpreter','none'
%                               'Color','m','Fontweight','bold'
%
% Output:   hText       - handle vector to text labels
%
% Example 1:  Rotate existing XTickLabels at their current position by 90°
%    xticklabel_rotate
%
% Example 2:  Rotate existing XTickLabels at their current position by 45° and change
% font size
%    xticklabel_rotate([],45,[],'Fontsize',14)
%
% Example 3:  Set the positions of the XTicks and rotate them 90°
%    figure;  plot([1960:2004],randn(45,1)); xlim([1960 2004]);
%    xticklabel_rotate([1960:2:2004]);
%
% Example 4:  Use text labels at XTick positions rotated 45° without tex interpreter
%    xticklabel_rotate(XTick,45,NameFields,'interpreter','none');
%
% Example 5:  Use text labels rotated 90° at current positions
%    xticklabel_rotate([],90,NameFields);
%
% Example 6:  Multiline labels
%    figure;plot([1:4],[1:4])
%    axis([0.5 4.5 1 4])
%    xticklabel_rotate([1:4],45,{{'aaa' 'AA'};{'bbb' 'AA'};{'ccc' 'BB'};{'ddd' 'BB'}})
%
% Note : you can not RE-RUN xticklabel_rotate on the same graph.
%



% This is a modified version of xticklabel_rotate90 by Denis Gilbert
% Modifications include Text labels (in the form of cell array)
%                       Arbitrary angle rotation
%                       Output of text handles
%                       Resizing of axes and title/xlabel/ylabel positions to maintain same overall size
%                           and keep text on plot
%                           (handles small window resizing after, but not well due to proportional placement with
%                           fixed font size. To fix this would require a serious resize function)
%                       Uses current XTick by default
%                       Uses current XTickLabel is different from XTick values (meaning has been already defined)

% Brian FG Katz
% bfgkatz@hotmail.com
% 23-05-03
% Modified 03-11-06 after user comment
%	Allow for exisiting XTickLabel cell array
% Modified 03-03-2006
%   Allow for labels top located (after user comment)
%   Allow case for single XTickLabelName (after user comment)
%   Reduced the degree of resizing
% Modified 11-jun-2010
%   Response to numerous suggestions on MatlabCentral to improve certain
%   errors.
% Modified 23-sep-2014
%   Allow for mutliline labels


% Other m-files required: cell2mat
% Subfunctions: none
% MAT-files required: none
%
% See also: xticklabel_rotate90, TEXT,  SET

% Based on xticklabel_rotate90
%   Author: Denis Gilbert, Ph.D., physical oceanography
%   Maurice Lamontagne Institute, Dept. of Fisheries and Oceans Canada
%   email: gilbertd@dfo-mpo.gc.ca  Web: http://www.qc.dfo-mpo.gc.ca/iml/
%   February 1998; Last revision: 24-Mar-2003

% check to see if xticklabel_rotate has already been here (no other reason for this to happen)
if isempty(get(gca,'XTickLabel')),
    error('xticklabel_rotate : can not process, either xticklabel_rotate has already been run or XTickLabel field has been erased')  ;
end

% if no XTickLabel AND no XTick are defined use the current XTickLabel
%if nargin < 3 & (~exist('XTick') | isempty(XTick)),
% Modified with forum comment by "Nathan Pust" allow the current text labels to be used and property value pairs to be changed for those labels
if (nargin < 3 || isempty(varargin{1})) & (~exist('XTick') | isempty(XTick)),
    xTickLabels = get(gca,'XTickLabel')  ; % use current XTickLabel
    if ~iscell(xTickLabels)
        % remove trailing spaces if exist (typical with auto generated XTickLabel)
        temp1 = num2cell(xTickLabels,2)         ;
        for loop = 1:length(temp1),
            temp1{loop} = deblank(temp1{loop})  ;
        end
        xTickLabels = temp1                     ;
    end
    varargin = varargin(2:length(varargin));
end

% if no XTick is defined use the current XTick
if (~exist('XTick') | isempty(XTick)),
    XTick = get(gca,'XTick')        ; % use current XTick
end

%Make XTick a column vector
XTick = XTick(:);

if ~exist('xTickLabels'),
    % Define the xtickLabels
    % If XtickLabel is passed as a cell array then use the text
    if (length(varargin)>0) & (iscell(varargin{1})),
        xTickLabels = varargin{1};
        varargin = varargin(2:length(varargin));
    else
        xTickLabels = num2str(XTick);
    end
end

if length(XTick) ~= length(xTickLabels),
    error('xticklabel_rotate : must have same number of elements in "XTick" and "XTickLabel"')  ;
end

%Set the Xtick locations and set XTicklabel to an empty string
set(gca,'XTick',XTick,'XTickLabel','')

if nargin < 2,
    rot = 90 ;
end

% Determine the location of the labels based on the position
% of the xlabel
hxLabel = get(gca,'XLabel');  % Handle to xlabel
xLabelString = get(hxLabel,'String');

% if ~isempty(xLabelString)
%    warning('You may need to manually reset the XLABEL vertical position')
% end

set(hxLabel,'Units','data');
xLabelPosition = get(hxLabel,'Position');
y = xLabelPosition(2);

%CODE below was modified following suggestions from Urs Schwarz
y=repmat(y,size(XTick,1),1);
% retrieve current axis' fontsize
fs = get(gca,'fontsize');

if ~iscell(xTickLabels)
    % Place the new xTickLabels by creating TEXT objects
    hText = text(XTick, y, xTickLabels,'fontsize',fs);
else
    % Place multi-line text approximately where tick labels belong
    for cnt=1:length(XTick),
        hText(cnt) = text(XTick(cnt),y(cnt),xTickLabels{cnt},...
            'VerticalAlignment','top', 'UserData','xtick');
    end
end

% Rotate the text objects by ROT degrees
%set(hText,'Rotation',rot,'HorizontalAlignment','right',varargin{:})
% Modified with modified forum comment by "Korey Y" to deal with labels at top
% Further edits added for axis position
xAxisLocation = get(gca, 'XAxisLocation');
if strcmp(xAxisLocation,'bottom')
    set(hText,'Rotation',rot,'HorizontalAlignment','right',varargin{:})
else
    set(hText,'Rotation',rot,'HorizontalAlignment','left',varargin{:})
end

% Adjust the size of the axis to accomodate for longest label (like if they are text ones)
% This approach keeps the top of the graph at the same place and tries to keep xlabel at the same place
% This approach keeps the right side of the graph at the same place

set(get(gca,'xlabel'),'units','data')           ;
labxorigpos_data = get(get(gca,'xlabel'),'position')  ;
set(get(gca,'ylabel'),'units','data')           ;
labyorigpos_data = get(get(gca,'ylabel'),'position')  ;
set(get(gca,'title'),'units','data')           ;
labtorigpos_data = get(get(gca,'title'),'position')  ;

set(gca,'units','pixel')                        ;
set(hText,'units','pixel')                      ;
set(get(gca,'xlabel'),'units','pixel')          ;
set(get(gca,'ylabel'),'units','pixel')          ;
% set(gca,'units','normalized')                        ;
% set(hText,'units','normalized')                      ;
% set(get(gca,'xlabel'),'units','normalized')          ;
% set(get(gca,'ylabel'),'units','normalized')          ;

origpos = get(gca,'position')                   ;

% textsizes = cell2mat(get(hText,'extent'))       ;
% Modified with forum comment from "Peter Pan" to deal with case when only one XTickLabelName is given.
x = get( hText, 'extent' );
if iscell( x ) == true
    textsizes = cell2mat( x ) ;
else
    textsizes = x;
end

largest =  max(textsizes(:,3))                  ;
longest =  max(textsizes(:,4))                  ;

laborigext = get(get(gca,'xlabel'),'extent')    ;
laborigpos = get(get(gca,'xlabel'),'position')  ;

labyorigext = get(get(gca,'ylabel'),'extent')   ;
labyorigpos = get(get(gca,'ylabel'),'position') ;
leftlabdist = labyorigpos(1) + labyorigext(1)   ;

% assume first entry is the farthest left
leftpos = get(hText(1),'position')              ;
leftext = get(hText(1),'extent')                ;
leftdist = leftpos(1) + leftext(1)              ;
if leftdist > 0,    leftdist = 0 ; end          % only correct for off screen problems

% botdist = origpos(2) + laborigpos(2)            ;
% newpos = [origpos(1)-leftdist longest+botdist origpos(3)+leftdist origpos(4)-longest+origpos(2)-botdist]
%
% Modified to allow for top axis labels and to minimize axis resizing
if strcmp(xAxisLocation,'bottom')
    newpos = [origpos(1)-(min(leftdist,labyorigpos(1)))+labyorigpos(1) ...
        origpos(2)+((longest+laborigpos(2))-get(gca,'FontSize')) ...
        origpos(3)-(min(leftdist,labyorigpos(1)))+labyorigpos(1)-largest ...
        origpos(4)-((longest+laborigpos(2))-get(gca,'FontSize'))]  ;
else
    newpos = [origpos(1)-(min(leftdist,labyorigpos(1)))+labyorigpos(1) ...
        origpos(2) ...
        origpos(3)-(min(leftdist,labyorigpos(1)))+labyorigpos(1)-largest ...
        origpos(4)-(longest)+get(gca,'FontSize')]  ;
end
set(gca,'position',newpos)                      ;

% readjust position of text labels after resize of plot
set(hText,'units','data')                       ;
for loop= 1:length(hText),
    set(hText(loop),'position',[XTick(loop), y(loop)])  ;
end

% adjust position of xlabel and ylabel
laborigpos = get(get(gca,'xlabel'),'position')  ;
set(get(gca,'xlabel'),'position',[laborigpos(1) laborigpos(2)-longest 0])   ;

% switch to data coord and fix it all
set(get(gca,'ylabel'),'units','data')                   ;
set(get(gca,'ylabel'),'position',labyorigpos_data)      ;
set(get(gca,'title'),'position',labtorigpos_data)       ;

set(get(gca,'xlabel'),'units','data')                   ;
labxorigpos_data_new = get(get(gca,'xlabel'),'position')  ;
set(get(gca,'xlabel'),'position',[labxorigpos_data(1) labxorigpos_data_new(2)])   ;


% Reset all units to normalized to allow future resizing
set(get(gca,'xlabel'),'units','normalized')          ;
set(get(gca,'ylabel'),'units','normalized')          ;
set(get(gca,'title'),'units','normalized')          ;
set(hText,'units','normalized')                      ;
set(gca,'units','normalized')                        ;

if nargout < 1,
    clear hText
end


function img2 = addborder(img1, t, c, stroke)
% ADDBORDER draws a border around an image
%
%    NEWIMG = ADDBORDER(IMG, T, C, S) adds a border to the image IMG with
%    thickness T, in pixels. C specifies the color, and should be in the
%    same format as the image itself. STROKE is a string indicating the
%    position of the border:
%       'inner'  - border is added to the inside of the image. The dimensions
%                  of OUT will be the same as IMG.
%       'outer'  - the border sits completely outside of the image, and does
%                  not obscure any portion of it.
%       'center' - the border straddles the edges of the image.
%
% Example:
%     load mandrill
%     X2 = addborder(X, 20, 62, 'center');
%     image(X2);
%     colormap(map);
%     axis off
%     axis image

%    Eric C. Johnson, 7-Aug-2008

% Input data validation
if nargin < 4
    error('MATLAB:addborder','ADDBORDER requires four inputs.');
end

if numel(c) ~= size(img1,3)
    error('MATLAB:addborder','C does not match the color format of IMG.');
end

% Ensure that C has the same datatype as the image itself.
% Also, ensure that C is a "depth" vector, rather than a row or column
% vector, in the event that C is a 3 element RGB vector.
c = cast(c, class(img1));
c = reshape(c,1,1,numel(c));

% Compute the pixel size of the new image, and allocate a matrix for it.
switch lower(stroke(1))
    case 'i'
        img2 = addInnerStroke(img1, t, c);
    case 'o'
        img2 = addOuterStroke(img1, t, c);
    case 'c'
        img2 = addCenterStroke(img1, t, c);
    otherwise
        error('MATLAB:addborder','Invalid value for ''stroke''.');
end




% Helper functions for each stroke type
function img2 = addInnerStroke(img1, t, c)

[nr1 nc1 d] = size(img1);

% Initially create a copy of IMG1
img2 = img1;

% Now fill the border elements of IMG2 with color C
img2(:,1:t,:)           = repmat(c,[nr1 t 1]);
img2(:,(nc1-t+1):nc1,:) = repmat(c,[nr1 t 1]);
img2(1:t,:,:)           = repmat(c,[t nc1 1]);
img2((nr1-t+1):nr1,:,:) = repmat(c,[t nc1 1]);


function img2 = addOuterStroke(img1, t, c)

[nr1 nc1 d] = size(img1);

% Add the border thicknesses to the total image size
nr2 = nr1 + 2*t;
nc2 = nc1 + 2*t;

% Create an empty matrix, filled with the border color.
img2 = repmat(c, [nr2 nc2 1]);

% Copy IMG1 to the inner portion of the image.
img2( (t+1):(nr2-t), (t+1):(nc2-t), : ) = img1;

function img2 = addCenterStroke(img1, t, c)

% Add an inner and outer stroke of width T/2
img2 = addInnerStroke(img1, floor(t/2), c);
img2 = addOuterStroke(img2, ceil(t/2), c);

