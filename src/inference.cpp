#include "inference.h"

Inference::Inference() : it(n) 
{
    detector = NanoDet("/home/px4vision/catkin/src/auav_2022_sample/object_tracking/src/nanodet.xml", "MYRIAD", 32);
    height = detector.input_size[0];
    width = detector.input_size[1];
    tracker = std::make_shared<BYTETracker>(10, 30);

    sub_rgb = it.subscribe("/drone/camera/color/image_raw", 1, &Inference::imageCallback, this);
    //sub_depth = it.subscribe("camera/depth/image_raw", 1, &Inference::imageCallback, this);
    pub_bbox = n.advertise<geometry_msgs::QuaternionStamped>("/drone/rover/bounding_box", 5);
    //pub_rel_pos = n.advertise<geometry_msgs::PointStamped>("rover/rel_pos", 5);
    old_bboxes.push_back(-1);
    old_bboxes.push_back(-1);
    old_bboxes.push_back(-1);
    old_bboxes.push_back(-1);

    old_point[0] = -1;
	old_point[1] = -1;
	old_point[2] = -1;

    model_in_size = cv::Size(width, height);
    new_bbox = null_bbox;
    old_bbox = null_bbox;
    new_cords = null_xyz;
    old_cords = null_xyz;
    ROS_INFO("Model Constructor finished");
}

Inference::~Inference()
{
}

void Inference::imageCallback(const sensor_msgs::ImageConstPtr &img_msg)
{
    using namespace cv;
    using namespace rs2;

    //BYTETracker tracker(10, 30);
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Mat color_mat = cv_ptr->image;

    Mat resized_img;
    object_rect effect_roi;
    resize_uniform(color_mat, resized_img, cv::Size(width, height), effect_roi);
    auto results = detector.detect(resized_img, conf_threshold, nms_threshold);
    //std::vector<float> bboxes = get_bboxes(color_mat, results, effect_roi);

    //Track objects with bytetrack
    vector<Object> bt_bboxes = convert_bytetrack(results, color_mat, effect_roi);
    vector<STrack> output_stracks = tracker->update(bt_bboxes);
    
    //Start output logic
    update_bbox_cord(output_stracks, old_bbox, new_bbox, old_cords, new_cords);

    /*float point[3];
    float mean_depth = 0;

    if (bboxes.size() == 0)
    {
        bboxes = old_bboxes;
        point[0] = old_point[0];
        point[1] = old_point[1];
        point[2] = old_point[2];

        count++;
        if (count > 30)
        {
            bboxes[0] = -1;
            bboxes[1] = -1;
            bboxes[2] = -1;
            bboxes[3] = -1;

            point[0] = -1;
            point[1] = -1;
            point[2] = -1;
        }
    }
    else
    {
        old_bboxes = bboxes;
        count = 0;
    }*/

    ROS_INFO("%f, %f, %f, %f, %d, %d\n", new_bbox.x1, new_bbox.y1, new_bbox.x2, new_bbox.y2, color_mat.cols, color_mat.rows);
    geometry_msgs::Quaternion msg;
    msg.x = new_bbox.x1; // x of top left
    msg.y = new_bbox.y1; // y of top left
    msg.z = new_bbox.x2-new_bbox.x1; // width
    msg.w = new_bbox.y2-new_bbox.y1; // height
    geometry_msgs::QuaternionStamped stamped_msg;
    stamped_msg.header = std_msgs::Header();
    stamped_msg.quaternion = msg;
    pub_bbox.publish(stamped_msg);
}

void Inference::update_bbox_cord(vector<STrack> stracks, bbox_tlbr& old_bbox, bbox_tlbr& new_bbox, xyz& old_cords, xyz& new_cords)
{
    //Output last detection if within 30 frames, else output null_bbox and null_xyz
    float max_conf = 0;
    new_bbox = null_bbox;
    new_cords = null_xyz;

    //Output last detection if within 30 frames
    if (stracks.size() == 0){
        if(elapsed_frames > 30){
            return;
        }
        ++elapsed_frames;
        new_bbox = old_bbox;
        new_cords = old_cords;
        return;
    }

    //Find max conf bbox from detections
    for (size_t i = 0; i < stracks.size(); i++)
    {
        if (stracks[i].score > max_conf){
            elapsed_frames = 0;
            max_conf = stracks[i].score;

            new_bbox.x1 = stracks[i].tlbr[0];
            new_bbox.y1 = stracks[i].tlbr[1];
            new_bbox.x2 = stracks[i].tlbr[2];
            new_bbox.y2 = stracks[i].tlbr[3];

            old_bbox = new_bbox;
            old_cords = new_cords;
        }
    }
    return;
}

int Inference::resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size, object_rect &effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst)
    {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst)
    {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else
    {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w)
    {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        for (int i = 0; i < dst_h; i++)
        {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h)
    {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else
    {
        printf("error\n");
    }
    return 0;
}

std::vector<float> Inference::get_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes, object_rect effect_roi)
{
    cv::Mat image = bgr.clone();

    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;
    std::vector<float> boundingboxes;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo &bbox = bboxes[i];

        int point = (bbox.x1 - effect_roi.x) * width_ratio;
        boundingboxes.push_back(point);
        point = (bbox.y1 - effect_roi.y) * height_ratio;
        boundingboxes.push_back(point);
        point = (bbox.x2 - effect_roi.x) * width_ratio;
        boundingboxes.push_back(point);
        point = (bbox.y2 - effect_roi.y) * height_ratio;
        boundingboxes.push_back(point);
        boundingboxes.push_back(bbox.score * 100);
    }

    return boundingboxes;
}

vector<Object> Inference::convert_bytetrack(const std::vector<BoxInfo>& results, const cv::Mat& image, const object_rect effect_roi)
{
    //Converts Nanodet bbox to ByteTrack style tlwh with score and label
    static int num_dets = results.size();
    vector<Object> objects;
    objects.resize(num_dets);

    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;
    for(int i=0; i< results.size(); ++i){
        objects[i].rect.x = (results[i].x1 - effect_roi.x) * width_ratio;
        objects[i].rect.y = (results[i].y1 - effect_roi.y) * height_ratio;
        objects[i].rect.width = ((results[i].x2 - effect_roi.x) * width_ratio) - ((results[i].x1 - effect_roi.x) * width_ratio);
        objects[i].rect.height = ((results[i].y2 - effect_roi.y) * height_ratio) - ((results[i].y1 - effect_roi.y) * height_ratio);
        objects[i].prob = results[i].score;
        objects[i].label = results[i].label;
    }
    return objects;
}