#include "loop_detection.h"

LoopDetector::LoopDetector(){}


void LoopDetector::loadVocabulary(std::string voc_path)
{
    voc = new BriefVocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}

// 闭环检测
void LoopDetector::addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
	int loop_index = -1; // 最小的闭环候选帧索引
    if (flag_detect_loop)
    {
        // 寻找闭环候选帧
        loop_index = detectLoop(cur_kf, cur_kf->index);
    }
    else
    {
        // 加入到关键帧数据库
        addKeyFrameIntoVoc(cur_kf);
    }

    // check loop if valid using ransan and pnp
	if (loop_index != -1)
	{
        // 候选关键帧
        KeyFrame* old_kf = getKeyFrame(loop_index);

        // 通过特征匹配和投影匹配, 判断是否是真闭环
        if (cur_kf->findConnection(old_kf))
        {
            std_msgs::Float64MultiArray match_msg;
            match_msg.data.push_back(cur_kf->time_stamp);
            match_msg.data.push_back(old_kf->time_stamp);
            pub_match_msg.publish(match_msg); // 发布闭环信息
        }
	}

    // add keyframe
	keyframelist.push_back(cur_kf);
}

// 通过index得到KeyFrame
KeyFrame* LoopDetector::getKeyFrame(int index)
{
    list<KeyFrame*>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)   
    {
        if((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return NULL;
}

// 寻找闭环候选帧
int LoopDetector::detectLoop(KeyFrame* keyframe, int frame_index)
{
    // 1.put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        // 压缩图像, 便于显示
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        // 在图像上显示文本
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[frame_index] = compressed_image;
    }

    // 2.查询字典数据库db，得到与每一帧的相似度评分ret
    // 2.first query; then add this frame into database!
    QueryResults ret;
    db.query(keyframe->brief_descriptors, ret, 4, frame_index - 200);
    //printf("query time: %f", t_query.toc());
    //cout << "Searching for Image " << frame_index << ". " << ret << endl;
    // 添加当前关键帧到字典数据库db中
    db.add(keyframe->brief_descriptors);
    //printf("add feature time: %f", t_add.toc());
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    
    // 3.闭环检测可视化
    cv::Mat loop_result;
    if (DEBUG_IMAGE)
    {
        loop_result = compressed_image.clone();
        if (ret.size() > 0)
            putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    }
    // visual loop result 
    if (DEBUG_IMAGE)
    {
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            int tmp_index = ret[i].Id;
            auto it = image_pool.find(tmp_index);
            cv::Mat tmp_image = (it->second).clone();
            putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
            cv::hconcat(loop_result, tmp_image, loop_result); // 图像水平拼接
        }
    }

    // a good match with its nerghbour
    bool find_loop = false;
    if (ret.size() >= 1 && ret[0].Score > 0.05)
    {
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {          
                find_loop = true; // 有闭环候选帧
                
                if (DEBUG_IMAGE && 0)
                {
                    int tmp_index = ret[i].Id;
                    auto it = image_pool.find(tmp_index);
                    cv::Mat tmp_image = (it->second).clone();
                    putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
                    cv::hconcat(loop_result, tmp_image, loop_result);
                }
            }

        }
    }
    
    if (DEBUG_IMAGE)
    {
        cv::imshow("loop_result", loop_result);
        cv::waitKey(20);
    }
    
    // 前50个关键帧不检测闭环
    if (find_loop && frame_index > 50)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || ((int)ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        return min_index; // 最小的闭环候选帧索引
    }
    else
        return -1;

}

// 加入到关键帧数据库
void LoopDetector::addKeyFrameIntoVoc(KeyFrame* keyframe)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[keyframe->index] = compressed_image;
    }

    db.add(keyframe->brief_descriptors);
}

void LoopDetector::visualizeKeyPoses(double time_cur)
{
    if (keyframelist.empty() || pub_key_pose.getNumSubscribers() == 0)
        return;

    visualization_msgs::MarkerArray markerArray;

    int count = 0;
    int count_lim = 10;

    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "vins_world";
    markerNode.header.stamp = ros::Time().fromSec(time_cur);
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "keyframe_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;

    // 遍历KeyFrames
    for (list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin(); rit != keyframelist.rend(); ++rit)
    {
        if (count++ > count_lim)
            break;

        geometry_msgs::Point p;
        p.x = (*rit)->origin_vio_T.x();
        p.y = (*rit)->origin_vio_T.y();
        p.z = (*rit)->origin_vio_T.z();
        markerNode.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    pub_key_pose.publish(markerArray);
}