% -------------------------------------------------------------------------
%   Edge Detection Using Canny Algorithm.
%   Auther: Yongli Yan.
%   Mail: yanyongli@ime.ac.cn
%   Date: 2017.08.01.
%   The direction of Sobel operator.
%   ^(y)
%   |
%   |
%   |
%   0--------->(x)
%   Direction of Gradient:
%               3   2   1
%               0   P   0
%               1   2   3
%   P = Current Point.
%               NW  N  NE
%               W   P   E
%               SW  S  SE
%   Point Index:
%               f(x-1,y-1)      f(x-1,y)    f(x-1,y+1)
%               f(x,  y-1)      f(x,  y)    f(x,  y+1)
%               f(x+1,y-1)      f(x+1,y)    f(x+1,y+1)
%   Parameters:
%   percentOfPixelsNotEdges: Used for selecting thresholds.
%   thresholdRatio: Low thresh is this fraction of the high.
% -------------------------------------------------------------------------
function imgCanny = edge_canny(I,gaussDim,sigma,percentOfPixelsNotEdges,thresholdRatio)
    %% Gaussian smoothing filter.
    m = gaussDim(1);
    n = gaussDim(2);
    if mod(m,2) == 0 || mod(n,2) == 0
        error('The dimensionality of Gaussian must be odd!');
    end
    % Generate gaussian convolution kernel.
    gaussKernel = fspecial('gaussian', [m,n], sigma);
    % Image edge copy.
    [m,n] = size(gaussKernel);
    [row,col,dim] = size(I);
    if dim > 1
        imgGray = rgb2gray(I);
    else
        imgGray = I;
    end
    imgCopy = imgReplicate(imgGray,(m-1)/2,(n-1)/2);
    % Gaussian smoothing filter.
    imgData = zeros(row,col);
    for ii = 1:row
        for jj = 1:col
            window = imgCopy(ii:ii+m-1,jj:jj+n-1);
            GSF = window.*gaussKernel;
            imgData(ii,jj) = sum(GSF(:));
        end
    end
    %% Calculate the gradient values for each pixel.
    % Sobel operator.
    dgau2Dx = [-1 0 1;-2 0 2;-1 0 1];
    dgau2Dy = [1 2 1;0 0 0;-1 -2 -1];
    [m,n] = size(dgau2Dx);
    % Image edge copy.
    imgCopy = imgReplicate(imgData,(m-1)/2,(n-1)/2);
    % To store the gradient and direction information.
    gradx = zeros(row,col);
    grady = zeros(row,col);
    gradm = zeros(row,col);
    dir = zeros(row,col); % Direction of gradient.
    % Calculate the gradient values for each pixel.
    for ii = 1:row
        for jj = 1:col
            window = imgCopy(ii:ii+m-1,jj:jj+n-1);
            dx = window.*dgau2Dx;
            dy = window.*dgau2Dy;
            dx = dx'; % Make the sum more accurate.
            dx = sum(dx(:));
            dy = sum(dy(:));
            gradx(ii,jj) = dx;
            grady(ii,jj) = dy;
            gradm(ii,jj) = sqrt(dx^2 + dy^2);
            % Calculate the angle of the gradient.
            theta = atand(dy/dx) + 90; % 0~180.
            % Determine the direction of the gradient.
            if (theta >= 0 && theta < 45)
                dir(ii,jj) = 2;
            elseif (theta >= 45 && theta < 90)
                dir(ii,jj) = 3;
            elseif (theta >= 90 && theta < 135)
                dir(ii,jj) = 0;
            else
                dir(ii,jj) = 1;
            end
        end
    end
    % Normalize for threshold selection.
    magMax = max(gradm(:));
    if magMax ~= 0
        gradm = gradm / magMax;
    end
    %% Plot 3D gradient graph.
    % [xx, yy] = meshgrid(1:col, 1:row);
    % figure;
    % surf(xx,yy,gradm);
    %% Threshold selection.
    counts = imhist(gradm, 64);
    highThresh = find(cumsum(counts) > percentOfPixelsNotEdges*row*col,1,'first') / 64;
    lowThresh = thresholdRatio*highThresh;
    %% Non-Maxima Suppression(NMS) Using Linear Interpolation.
    gradmCopy = zeros(row,col);
    imgBW = zeros(row,col);
    for ii = 2:row-1
        for jj = 2:col-1
            E =  gradm(ii,jj+1);
            S =  gradm(ii+1,jj);
            W =  gradm(ii,jj-1);
            N =  gradm(ii-1,jj);
            NE = gradm(ii-1,jj+1);
            NW = gradm(ii-1,jj-1);
            SW = gradm(ii+1,jj-1);
            SE = gradm(ii+1,jj+1);
            % Linear interpolation.
            % dy/dx = tan(theta).
            % dx/dy = tan(90-theta).
            gradValue = gradm(ii,jj);
            if dir(ii,jj) == 0
                d = abs(grady(ii,jj)/gradx(ii,jj));
                gradm1 = E*(1-d) + NE*d;
                gradm2 = W*(1-d) + SW*d;
            elseif dir(ii,jj) == 1
                d = abs(gradx(ii,jj)/grady(ii,jj));
                gradm1 = N*(1-d) + NE*d;
                gradm2 = S*(1-d) + SW*d;
            elseif dir(ii,jj) == 2
                d = abs(gradx(ii,jj)/grady(ii,jj));
                gradm1 = N*(1-d) + NW*d;
                gradm2 = S*(1-d) + SE*d;
            elseif dir(ii,jj) == 3
                d = abs(grady(ii,jj)/gradx(ii,jj));
                gradm1 = W*(1-d) + NW*d;
                gradm2 = E*(1-d) + SE*d;
            else
                gradm1 = highThresh;
                gradm2 = highThresh;
            end
            % Non-Maxima Suppression.
            if gradValue >= gradm1 && gradValue >= gradm2
                if gradValue >= highThresh
                    imgBW(ii,jj) = 1;
                    gradmCopy(ii,jj) = highThresh;
                elseif gradValue >= lowThresh
                    gradmCopy(ii,jj) = lowThresh;
                else
                    gradmCopy(ii,jj) = 0;
                end
            else
                gradmCopy(ii,jj) = 0;
            end
        end
    end
    %% High-Low threshold detection.Double-Threshold.
    % If the 8 pixels around the low threshold point have high threshold, then
    % the low threshold pixel should be retained.
    for ii = 2:row-1
        for jj = 2:col-1
            if gradmCopy(ii,jj) == lowThresh
                neighbors = [...
                    gradmCopy(ii-1,jj-1),   gradmCopy(ii-1,jj), gradmCopy(ii-1,jj+1),...
                    gradmCopy(ii,  jj-1),                       gradmCopy(ii,  jj+1),...
                    gradmCopy(ii+1,jj-1),   gradmCopy(ii+1,jj), gradmCopy(ii+1,jj+1)...
                    ];
                if ~isempty(find(neighbors) == highThresh)
                    imgBW(ii,jj) = 1;
                end
            end
        end
    end
    imgCanny = logical(imgBW);
    end
    %% Local functions. Image Replicate.
    function imgRep = imgReplicate(I,rExt,cExt)
    [row,col] = size(I);
    imgCopy = zeros(row+2*rExt,col+2*cExt);
    % 4 edges and 4 corners pixels.
    top = I(1,:);
    bottom = I(row,:);
    left = I(:,1);
    right = I(:,col);
    topLeftCorner = I(1,1);
    topRightCorner = I(1,col);
    bottomLeftCorner = I(row,1);
    bottomRightCorner = I(row,col);
    % The coordinates of the oroginal image after the expansion in the new graph.
    topLeftR = rExt+1;
    topLeftC = cExt+1;
    bottomLeftR = topLeftR+row-1;
    bottomLeftC = topLeftC;
    topRightR = topLeftR;
    topRightC = topLeftC+col-1;
    bottomRightR = topLeftR+row-1;
    bottomRightC = topLeftC+col-1;
    % Copy original image and 4 edges.
    imgCopy(topLeftR:bottomLeftR,topLeftC:topRightC) = I;
    imgCopy(1:rExt,topLeftC:topRightC) = repmat(top,[rExt,1]);
    imgCopy(bottomLeftR+1:end,bottomLeftC:bottomRightC) = repmat(bottom,[rExt,1]);
    imgCopy(topLeftR:bottomLeftR,1:cExt) = repmat(left,[1,cExt]);
    imgCopy(topRightR:bottomRightR,topRightC+1:end) = repmat(right,[1,cExt]);
    % Copy 4 corners.
    for ii = 1:rExt
        for jj = 1:cExt
            imgCopy(ii,jj) = topLeftCorner;
            imgCopy(ii,jj+topRightC) = topRightCorner;
            imgCopy(ii+bottomLeftR,jj) = bottomLeftCorner;
            imgCopy(ii+bottomRightR,jj+bottomRightC) = bottomRightCorner;
        end
    end
    imgRep = imgCopy;
    end
    %% End of file.