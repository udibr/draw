
Sketch
------

[Eitz et al](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) asked non-expert humans to sketch objects of a given category and gather 20,000 unique sketches evenly distributed over 250 object categories.

    @article{eitz2012hdhso,
        author={Eitz, Mathias and Hays, James and Alexa, Marc},
        title={How Do Humans Sketch Objects?},
        journal={ACM Trans. Graph. (Proc. SIGGRAPH)},
        year={2012},
        volume={31},
        number={4},
        pages = {44:1--44:10}
    }

The original sketch png files are converted to numpy npy files with 
[Sketch.ipynb](http://nbviewer.ipython.org/github/udibr/draw/blob/master/datasets/Sketch.ipynb)

The two npy files created should be placed in the directory binarized_sketch in fuel data directory

To learning and generating images works like this

    cd ../draw
    python train-draw.py --name=sketch --attention=2,5 --niter=64 --lr=3e-4 --epochs=100
    python sample.py --size=56 sketch-r2-w5-t64-enc256-dec256-z100-lr34_log_model.pkl
    convert -delay 5 -loop 1 samples-*.png sketch.gif