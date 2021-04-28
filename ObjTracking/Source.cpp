#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/types_c.h>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

int main() {

	vector<String> fn;
	vector<Mat> data;
	glob("data/objects/*.png", fn, true);
	cout << fn.size() << endl;
	for (size_t k = 0; k < fn.size(); ++k)
	{
		Mat im = imread(fn[k]);
		if (im.empty()) continue; //only proceed if sucsessful
		cvtColor(im, im, COLOR_BGR2GRAY);
		data.push_back(im);
	}

	//cout << data.size() << endl;

	vector<Mat> homographies;
	Mat frame, prev_frame, gray, prev_gray;

	vector<vector<KeyPoint>> best_keypoints_img, best_keypoints_obj;
	vector<vector<Point2f>> keypoints_frame_xy, prev_corners, prev_keypoints(data.size());
	vector<KeyPoint> all_keypoints;
	vector<KeyPoint> next_keypoints1;
	vector<uchar> status;
	vector<float> err;
	vector<vector<Point2f>> next_keypoints = keypoints_frame_xy;
	vector<Point2f> keypoints_frame_temp;
	vector<KeyPoint> keyponts_frame;
	vector<Scalar> colors = { Scalar(255,0,0),Scalar(0,0,255),Scalar(0,255,0),Scalar(0,200,200) };
	int count = 0;
	VideoCapture cap("video.mov");

	if (cap.isOpened())
	{
		for (;;)
		{

			cap >> frame;
			if (count == 0)
			{
				frame.copyTo(prev_frame);
				Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
				cvtColor(frame, gray, COLOR_BGR2GRAY);
				gray.copyTo(prev_gray);
				Ptr<SIFT> detector = SIFT::create();
				Ptr<DescriptorExtractor> extractor = SIFT::create();

				Mat img_extractor, snap_extractor;

				for (int k = 0; k < data.size(); k++)
				{
					vector<DMatch> matches;
					vector<DMatch> good_matches, best_matches;
					vector<KeyPoint> keypoints_1, keypoints_2, keypoints_temp1, keypoints_temp2;

					detector->detect(data[k], keypoints_1);
					detector->detect(gray, keypoints_2);
					extractor->compute(data[k], keypoints_1, img_extractor);
					extractor->compute(gray, keypoints_2, snap_extractor);
					matcher->match(img_extractor, snap_extractor, matches);

					double max_dist = 0; double min_dist = 100;

					//-- Quick calculation of max and min distances between keypoints
					for (int j = 0; j < matches.size(); j++)
					{
						double dist = matches[j].distance;
						if (dist < min_dist) min_dist = dist;
						if (dist > max_dist) max_dist = dist;
					}

					for (int p = 0; p < matches.size(); p++)
					{
						if (matches[p].distance <= max(3 * min_dist, 0.02))
						{
							good_matches.push_back(matches[p]);
						}
					}

					vector<Point2f> obj;
					vector<Point2f> scene;

					if (good_matches.size() >= 4)
					{
						for (int i = 0; i < good_matches.size(); i++)
						{
							obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
							scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
						}
					}

					Mat H;
					Mat mask;
					H = findHomography(obj, scene, RANSAC, 3.0, mask);
					homographies.push_back(H);
					//cout << good_matches.size() << endl;
					for (int r = 0; r < good_matches.size(); r++)
					{
						if ((int)mask.at<uchar>(r, 0) != 0)
						{
							best_matches.push_back(good_matches[r]);
							keypoints_temp2.push_back(keypoints_2[good_matches[r].trainIdx]);

						}
					}
					best_keypoints_img.push_back(keypoints_temp2);

					//drawMatches(data[k], keypoints_temp1, gray, keypoints_temp2, good_matches, img_matches, colors[k], colors[k], vector<char>(), DrawMatchesFlags::DEFAULT);
				}
				for (int k = 0; k < data.size(); k++)
				{
					vector<Point2f> scene_corners;
					vector<Point2f> obj_corners;
					KeyPoint::convert(best_keypoints_img[k], prev_keypoints[k]);
					obj_corners.push_back(Point2f(0, 0));
					obj_corners.push_back(Point2f(data[k].cols - 1, 0));
					obj_corners.push_back(Point2f(data[k].cols - 1, data[k].rows - 1));
					obj_corners.push_back(Point2f(0, data[k].rows - 1));

					perspectiveTransform(obj_corners, scene_corners, homographies[k]);
					prev_corners.push_back(scene_corners);
					//drawing the rectangle
					for (int i = 0; i < 4; i++)
					{
						drawKeypoints(frame, best_keypoints_img[k], frame, colors[k]);
						line(frame, scene_corners[i], scene_corners[(i + 1) % 4], colors[k], 4);
					}
				}



				imshow("Result", frame);
				count++;
				cout << "Premere tasto per andare avanti " << endl;
				waitKey(0);
			}
			else
			{
				if (!frame.empty())
				{

					vector<uchar> status;
					vector<float> err;
					vector<Mat> mask(data.size());
					vector<vector<Point2f>> keypoints_aux(data.size()), next_keypoints(data.size());
					vector<vector<KeyPoint>> keypoints_frame(data.size());
					vector<vector<Point2f>> curr_corners(data.size());
					cvtColor(frame, gray, COLOR_BGR2GRAY);
					cvtColor(prev_frame, prev_gray, COLOR_BGR2GRAY);



					//tracking
					for (int k = 0; k < data.size(); k++)
					{

						calcOpticalFlowPyrLK(prev_gray, gray, prev_keypoints[k], next_keypoints[k], status, err);
						homographies[k] = findHomography(prev_keypoints[k], next_keypoints[k], RANSAC, 3.0, mask[k]);

						perspectiveTransform(prev_corners[k], curr_corners[k], homographies[k]);

						for (int r = 0; r < mask[k].rows; r++)
							if ((int)mask[k].at<uchar>(r, 0) != 0)
								keypoints_aux[k].push_back(next_keypoints[k][r]);

						KeyPoint::convert(keypoints_aux[k], keypoints_frame[k]);
						drawKeypoints(frame, keypoints_frame[k], frame, colors[k]);

						for (int i = 0; i < 4; i++)
						{
							line(frame, curr_corners[k][i], curr_corners[k][(i + 1) % 4], colors[k], 4);

						}
						prev_keypoints[k] = next_keypoints[k];
						prev_corners[k] = curr_corners[k];

					}

					imshow("Result", frame);
					waitKey(1);
					frame.copyTo(prev_frame);
					cap >> frame;
				}
				else
				{
					cout << "Premere un tasto per uscire" << endl;
					waitKey(0);
					return -1;
				}
			}
		}
	}
}