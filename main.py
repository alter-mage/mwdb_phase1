import download_dataset
import elbp
import hog
import moments_of_color

images=download_dataset.get_image_array()
moc_df=moments_of_color.get_cm_df(images)
elbp_df=elbp.get_elbp_df(images)
hog_df=hog.get_hog_df(images)

print('here')