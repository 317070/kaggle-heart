1. Summary

4-6 sentences summarizing the most important aspects of your model and analysis, such as:

How you spent your time (Eg. what proportion on feature engineering vs. machine learning?)
Your most important insight into the data
The most important features
The training method(s) you used
The tool(s) you used

We spent most on our time on the machine learning part, and quite some effort on the preprocessing of the data. Say in the end, time was split 50-50.
We reckon this data too dirty to do meaningful extrapolations for medical applications. There are so many weird stuff going on. The most important feature is the chamber of the heart itself.
We used deep learning approaches, in which we designed custom layers to add as much a priori information in the model as possible.
For this, we used lasagne, on top of Theano, on top of cudnn, on top of cuda to run our models.

2. Features Selection / Extraction

What were the most important features? *
How did you select features?
Did you make any important feature transformations?
Did you find any interesting interactions between features?
Did you use external data?
*We suggest you provide:

a variable importance plot showing the 10-20 most important features and
partial plots for the 3-5 most important features
If this is not possible, you should provide a list of the most important features.

We did not extract the most important features. We just tried to zoom in on the relevant chamber, and and let the deep learning do its work.
There was no feature selection. If a model does well on our own validation set, it is a good model.
We did add top-layers which tell a deep learning network how to calculate volumes from a set of slices.
No external data was used.

3. Training Method(s)

What training methods did you use?
Did you ensemble the models?
If you did ensemble, how did you weight the different models?

We used a gradient descent based method (Adam), which is common in deep learning approaches.

We did ensemble our models, we weight them on our own validation set (~16% of our training set). Not every model needs to predict every patient either.
Models can choose not to predict a patient, because they know they'll have problems. For instance, when a patient has a small number of sax slices, or a patient has no 4ch slice.
When ensembling, we check per patient if all models agree in the model agree enough with the final average distribution. If not, they are thrown out of the ensemble and a new ensemble is trained.

4. Simple Features and Methods

Many customers are happy to trade off model performance for simplicity. With this in mind:

Is there a subset of features that would get 90-95% of your final performance? Which features? *
What model that was most important? *
What would the simplified model score?
*Try and restrict your simple model to fewer than 10 features and one training method.

Yes, we have single models which achieve a score of crps 0.1090 on our own validation set. It is a meta-model, and is quite simple to understand and implement.

5. Interesting findings

What was the most important trick you used?
What do you think set you apart from others in the competition?
Did you find any interesting relationships in the data that don't fit in the sections above?

The most important trick was to use preprocessing and classic computer vision approaches to select the relevant part of the image, and zoom in on that part of the image.
This way, the learning algorithm can learn faster, and needs less training examples (the number of which were quite low in this competition).

6. Background on you/your team

If part of a team, please answer these questions for each team member. For larger teams (3+), please give shorter responses.

What was your background prior to entering this challenge?
Did you have any prior experience that helped you succeed in this competition?
What made you decide to enter this competition?
How much time did you spend on the competition?
If part of a team, how did you decide to team up?
If you competed as part of a team, who did what?

We are four people from the data science lab of Ghent University.

Appendix

This section is for a technical audience who are trying to run your solution. Please make sure your code is well commented.

A1. Dependencies

List of all dependencies, libraries, functions, packages or other third-party code used to generate your solution.

We ran this code on 5 computers:
Computer 1:
32GB RAM
GPU0: Tesla 12GB
GPU1: Tesla 12GB

Computer 2:
32GB RAM
GPU1: 680 4GB
GPU0: 680 4GB

Computer 3:
32GB RAM
GPU0: 680 2GB
GPU1: 680 4GB

Computer 4:
18GB RAM
GPU0: Tesla 12GB
GPU1: Tesla 12GB

Computer 5:
32GB RAM
GPU0: TITAN X 12GB
GPU1: 980 4GB
GPU2: 980 4GB

Each of these computers has a linux Ubuntu 14.04, together with cuda 7.5, cudnn4, Theano and Lasagne installed.
You also need recent version of scikit-learn, scikit-image, numpy, scipy, blz, pydicom.

A2. How To Generate the Solution (aka README file)

We refer to the RUNME.md file for a complete explanation.

A3. References

Citations to references, websites, blog posts, and external sources of information where appropriate.