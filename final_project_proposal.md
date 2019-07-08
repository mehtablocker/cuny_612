Final Project Proposal
================

## Using a linear regression to calculate user and item biases in a recommender system

#### Intro

Subtracting a “baseline” from user-item ratings is a common first step
in the building of recommender systems. This is usually done by
calculating a global average (i.e., “global bias”) plus an average for
each user (i.e., “user bias”) and an average for each item (i.e., “item
bias”).

However, using averages strikes me as overly simplistic. They might be
fine if the user-item matrix were more densely populated. But with such
a sparse matrix, how do we know that a user’s average is not the result
of consuming a few particularly good (or bad) items?

A seemingly better procedure for obtaining biases would be to run a
regression where all users and items are the predictor variables and the
ratings are the response variable. Then the coefficients for each user
and item would be the “user bias” and “item bias”, and the intercept
would be the “global bias”.

#### Implementation

One way of achieving such a regression would be to do a form of “one-hot
encoding” for every user, item, and rating. For example, say we use the
MovieLens data and see that Joe gave Toy Story a 5 rating. Rather than
having 5 in the cell of the user-item matrix that corresponds to Joe and
Toy Story, we have a matrix where the columns are every user and every
movie, and one additional column is the rating. In that case our first
row would have a zero in every cell except a 1 for Joe, a 1 for Toy
Story, and a 5 for Rating. And then the same idea for the next row
(rating.)

We could then run a regression where the Rating column is the response
variable and all of the other columns are the predictors. The result of
the regression would give a coefficient for every user and item, as well
as an intercept.

#### Considerations

Computing power is a concern when running a regression of this size.
Even with the moderately sized MovieLens 100k ratings (our dataset of
choice for this project), we will be computing the coefficients for over
25,000 predictor variables. This is computationally expensive.

To mitigate this expense we can use a linear algebra trick. Instead of
running a regression with our one-hot encode matrix A and our ratings
vector b, we can left-multiply both of these by A<sup>T</sup> and then
directly solve for x. This is a much cheaper computational task. To make
this work we will need to employ a couple other tricks (to account for
the intercept as well as the fact that the A<sup>T</sup> A matrix is
singular), but overall it is a fairly easy procedure.

#### Evaluating this methodology

Once armed with our coefficients and intercept, i.e., the new bias
values, we can then use these to complete the building of the
recommender system. The RMSE of the resulting prediction matrix
(performed on held-out Test data) can be compared to the RMSE of the
prediction matrix obtained using standard biases (i.e., simple
averages). My suspicion is that the regression-based biases will result
in more accurate predictions.
