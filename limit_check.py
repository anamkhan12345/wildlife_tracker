
detection_limit = 1000
# By detection split
single_split = int(.6 * detection_limit)
mlt_split = int(.2 * detection_limit)
detectable_split = single_split + mlt_split
neg_limit = int(.2 * detection_limit)

small_total_split = int(.35 * detectable_split)
med_total_split = int(.45 * detectable_split)
big_total_split = int(.2 * detectable_split)

# Combined detection and size split
single_small_limit = int(.35 * single_split)
single_med_limit = int(.45  * single_split)
single_big_limit = int(.2  * single_split)

mlt_small_limit = int(.35 * mlt_split)
mlt_med_limit = int(.45  * mlt_split)
mlt_big_limit = int(.2 * mlt_split)

# Check math 
check_total = (single_small_limit + single_med_limit + single_big_limit  + 
                mlt_small_limit + mlt_med_limit + mlt_big_limit + 
                neg_limit)

check_single = single_small_limit + single_med_limit + single_big_limit
check_mlt = mlt_small_limit + mlt_med_limit + mlt_big_limit


if check_single != single_split:
    print("**************")
    print("issue with single split")
    print(f"Small Split: {single_small_limit}, Med Split: {single_med_limit}, Big Split: {single_big_limit}, Total: {single_small_limit + single_med_limit + single_big_limit}")
    print(f"Single Total: {single_split}")

if check_mlt != mlt_split:
    print("**************")
    print("issue with Mltpl split")
    print(f"Small Split: {mlt_small_limit}, Med Split: {mlt_med_limit}, Big Split: {mlt_big_limit}")
    print(f"Single Total: {mlt_split}")

if check_total != detection_limit:
    if (single_small_limit + mlt_small_limit) != (small_total_split):
        print("Something wrong with your SMALL splits")
        print(f"Single Split: {single_small_limit}, Mlt Split: {mlt_small_limit}, Total: {small_total_split}")
    elif (single_med_limit + mlt_med_limit) != (med_total_split):
        print("Something wrong with your MED splits")
        print(f"Single Split: {single_med_limit}, Mlt Split: {mlt_med_limit}, Total: {med_total_split}")
    elif (single_big_limit + mlt_big_limit) != (big_total_split):
        print("Something wrong with your BIG splits")
        print(f"Single Split: {single_big_limit}, Mlt Split: {mlt_big_limit}, Total: {big_total_split}")
    else:
        print("something is wrong but idk")
        print(f"Check Total: {check_total}, Detection Limit: {detection_limit}")
else:
    print("good calc!")
