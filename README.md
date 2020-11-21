# CS74 Extra Credit Assignment
## Goal
For this extra credit assignment, I built upon the group project that I had just completed, which you can find at the CS74-Project repository. I added two new models, an MLP Classifier and a Decision Tree Classifier. My goal was to determine whether we would have found any significant improvement by adding new types of classifiers to our final SVM.
## Implementation
Instead of creating one final SVM, I created four. One contained no new models, one contained the MLP Classifier, one contained the Decision Tree Classifier, and one contained both.
## Outcomes
Overall, I saw no change in weighted average f1 score in any of the four SVMs, as all of them produced an f1 of 0.84. For the most part, less than a hundred entries had their classifications changed by the updated models. This finding suggests that we selected an adequate number and type of models to best optimize our final SVM. It also suggests that while adding more models may not help performance, it also likely would not hurt, so long as the models are appropriate for the situation. A regression model, for example, might not work as well for this classification problem.
