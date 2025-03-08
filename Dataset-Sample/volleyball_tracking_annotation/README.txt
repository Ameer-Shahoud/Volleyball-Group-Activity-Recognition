## Paper   
@inproceedings{SendoMVA2019Heatmapping,
  author    = {Kohei Sendo and Norimichi Ukita},
  title     = {Heatmapping of People Involved in Group Activities},
  booktitle = {16th International Conference on Machine Vision Applications (MVA)},
  year      = {2019}
}

##Original Dataset 
You can download the original dataset from the following page:
https://github.com/mostafa-saad/deep-activity-rec#dataset
@inproceedings{msibrahiCVPR16deepactivity,
  author    = {Mostafa S. Ibrahim and Srikanth Muralidharan and Zhiwei Deng and Arash Vahdat and Greg Mori},
  title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition.},
  booktitle = {2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

We used VATIC for manually annotating the data:
https://github.com/cvondrick/vatic

## Tracking and grouping
Grouping is a set of people involved in a primary group activity.
The players involved in each group acitivity are defined as follows:
[Pass]: Players who are trying an underhand pass independently of whether or not they are successful.
[Set]: Player who is doing an overhand pass and those who are going to spike the ball whether they are really trying or faking.
[Spike]: Players who are spiking and blocking. 
[Winpoint]: All players in the team that scores a point. This group activity is observed for a few seconds right after the score.

## Annotated data
We annotated 4830 sequences that are consisted of 55 videos as follows:
- We annotated the bounding-boxes of all players from 11 to 30 frames (10 images before a target frame, the target frame, and 9 frames after the target frame).
- Each video directory has a text file that contains framewise annotations
- Each annotation line consists of the following components: {player ID} {xmin} {ymin} {xmax} {ymax} {frame ID} {lost} {grouping} {generated} {individual action label}
- player ID
- xmin, ymin, xmax, ymax: The bounding box of this player.
- frame ID
- lost: If 1, the annotated bounding box is outside of the field of view.
- grouping: If 1, this player is involved in a primary group activity.
- generated: If 1, the bounding box was automatically interpolated.
- individual action label: The individual action label of each player, which was given in the original annotation of the Volley ball dataset.
