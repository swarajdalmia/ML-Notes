# Suggestions for training ML models 

[Source](https://www.wandb.com/podcast/josh-tobin)

- there's a tendency to get all this excitement around neural network architectures and the latest and greatest state-of-the-art model on image net. And so I think people tend to overthink the question of architecture selection and selection of all the other pieces around that; like what Optimizer are you using and things like that. But in reality, **I think when you're starting on a new project, the goal is to just choose a reasonable default and start there, even if it's not state-of-the-art**. And then once you've convinced yourself that everything around that is working, it's your data loading code and your training code and all that stuff, then you can gradually move closer to a state-of-the-art architecture.
-  the first thing that I  recommend people do when you're training a new neural net for the first time is just make sure that you can first. I mean, first of all, just get the thing to run.
-  Then the next thing that I think you usually want to do is try to overfit a really small amount of data; like a single batch of data.
- So typically, what I would do next is I would move from a single batch of data to a smaller or more simplified version of the dataset that I was working with. So maybe it's like, I don't know, maybe it's a million images, you only take a thousand or ten thousand of them to start out with.  
