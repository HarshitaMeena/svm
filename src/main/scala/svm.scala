import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.indexing.{BooleanIndexing, NDArrayIndex}
//import org.nd4j.linalg.indexing.BooleanIndexing

class SVM(ndims: Int, w_init: String = "zeros", decay: Double = 0.001) {
  var svm_weights: INDArray = _
  if (w_init == "zeros") {
    svm_weights = Nd4j.zeros(ndims, 1)
  } else if (svm_weights == "ones") {
    svm_weights = Nd4j.ones(ndims, 1)
  }
  var decay_factor = decay

  def shuffleData(x: INDArray, y: INDArray): INDArray = {
    val stackedData = Nd4j.hstack(x,y)
    Nd4j.shuffle(stackedData, 1)
    //print(combineddata.getRow(0))
    stackedData
  }

  /**
    * Forward operation for logistic models.
    * Performs the forward operation, and return probability score (sigmoid).
    * Args:
    * X: input dataset with a dimension of (# of samples, ndims+1)
    * Returns: probability score of (label == +1) for each sample
    * with a dimension of (# of samples,)
    */
  def forward(input: INDArray): INDArray = {
    var xTow = input.mmul(svm_weights)
    xTow
  }

  /**
    * Backward operation for logistic models.
    * Compute gradient according to the probability loss on lecture slides
    * Args:
    * X: input dataset with a dimension of (# of samples, ndims+1)
    * Y_true: dataset labels with a dimension of (# of samples,)
    * Returns: gradients of weights
    */
  def backward(f: INDArray, y_true: INDArray, input: INDArray): INDArray = {

    var fy = f.mul(y_true)
    BooleanIndexing.replaceWhere(fy, 1, Conditions.lessThan(1))
    BooleanIndexing.replaceWhere(fy, 0, Conditions.greaterThan(1))
    fy = fy.mul(y_true)
    //print(fy)
    var gradient  = input.mulColumnVector(fy)
    gradient = gradient.sum(0)
    gradient = gradient.transpose()
    gradient.muli(-1)
    //print(gradient.shape().toSeq, svm_weights.shape().toSeq)
    gradient.addi(svm_weights.mul(decay_factor))
    gradient
  }

  /**
    * Performs binary classification on input dataset.
    * Args:
    * X: input dataset with a dimension of (# of samples, ndims+1)
    * Returns: predicted label = +1/-1 for each sample
    * with a dimension of (# of samples,)
    */
  def classify(input: INDArray): INDArray = {
    var score = forward(input)
    BooleanIndexing.replaceWhere(score, -1.0, Conditions.lessThan(0))
    BooleanIndexing.replaceWhere(score, 1.0, Conditions.greaterThanOrEqual(0))
    //print(score)
    score
  }

  /**
    * train model with input dataset using gradient descent.
    * Args:
    * Y_true: dataset labels with a dimension of (# of samples,)
    * X: input dataset with a dimension of (# of samples, ndims+1)
    * learn_rate: learning rate for gradient descent
    * max_iters: maximal number of iterations
    * ......: append as many arguments as you want
    */
  def fit(y_true: INDArray, input: INDArray, learn_rate: Double, max_iters: Int, batchsize: Int): Unit = {

    var t1 = System.nanoTime
    var j = 1
    val totalbatches = math.ceil(input.rows / batchsize.toFloat).toInt
    var combinedData = shuffleData(input, y_true)
    for (i <- 0 to max_iters-1) {

      if (j == totalbatches+1) {
        combinedData = shuffleData(input, y_true)
        j = 1
      }
      //var t2 = System.nanoTime
      val subpart = combinedData.get(NDArrayIndex.interval((j - 1) * batchsize, math.min(j * batchsize, combinedData.rows)))
      val xbatch = subpart.get(NDArrayIndex.all(), NDArrayIndex.interval(0, subpart.columns - 1))
      val ybatch = subpart.get(NDArrayIndex.all(), NDArrayIndex.point(subpart.columns - 1))

      var f = forward(xbatch)
      //print(f)
      var gradient = backward(f, ybatch, xbatch)
      //print(gradient)
      svm_weights = svm_weights.add(gradient.mul(-learn_rate))
      j += 1
      //println(i, j, loss(y_true, input))
      //print(i, (System.nanoTime-t2)/1e9, xbatch.shape().toSeq)
    }
    val duration = (System.nanoTime - t1) / 1e9
    println("TOTAL TIME FOR PROCESSING for batchsize: ",  batchsize, " and iterations ", max_iters , " is ", duration)

  }

  /**
    *
    */
  def accuracy(y: INDArray, y_true: INDArray): Double = {
    var count = 0.0
    for (i <- 0 to y.size(0)-1) {
      if (y.getDouble(i,0) == y_true.getDouble(i,0)) {
        count += 1
      }
    }
    count/y.rows()
  }

  /**
    *
    */
  def loss(y_true: INDArray, input: INDArray): Double = {
    //println(lg_weights)
    var hinge_loss = 0.0d
    val f = forward(input)
    var fy = f.mul(y_true)
    BooleanIndexing.replaceWhere(fy, 1, Conditions.greaterThan(1))
    fy.subi(1)
    fy.muli(-1)
    hinge_loss += fy.sumNumber().doubleValue()
    var l2_loss = 0.5 * decay_factor * svm_weights.norm2Number().doubleValue() * svm_weights.norm2Number().doubleValue()

    l2_loss+hinge_loss
  }
}
