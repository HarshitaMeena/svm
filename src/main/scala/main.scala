
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.io.Source

object DataReader {

  def readFromFiles(labelfile: String, input: String): (INDArray, INDArray) = {
    var labels = Array[Double]()
    var images = Array[String]()
    var total = 0
    for (line <- Source.fromFile(labelfile).getLines) {
      //println(line)
      var m = line.split(",")
      images = images :+ m(0)
      labels = labels :+ m(1).toDouble
      total += 1
    }
    //labels.foreach(z => println(z))
    var y = Nd4j.create(labels, Array[Int](total, 1))//(labels, new int[]{total,1})
    //print(Nd4j.shape(y).toSeq)

    ////// READ IMAGES INTO INDARRAY
    /*
    var imagefile = "src/main/resources/image_data/"+images(0)+".jpg"
    var f = new File(imagefile)
    val loader = new NativeImageLoader()
    var allimage = loader.asMatrix(f)

    for (l <- images.slice(1,total)) {
      imagefile = "src/main/resources/image_data/"+l+".jpg"
      f = new File(imagefile)
      allimage = Nd4j.concat(0,allimage, loader.asMatrix(f))
    }
    print(Nd4j.shape(allimage).toSeq)//, allimage.getRow(0), allimage.getRow(1))
    */
    //var imagefile = "src/main/resources/file-1000"
    val recordReaderinput = new CSVRecordReader(0, ",")
    val pathsourceinput = new ClassPathResource(input).getFile
    val file1 = new FileSplit(pathsourceinput)
    recordReaderinput.initialize(file1)
    var concVal = recordReaderinput.next().toArray().map(_.toString).map(_.toFloat)
    var x = Nd4j.create(Array(concVal))
    while (recordReaderinput.hasNext()) {
      concVal = recordReaderinput.next().toArray().map(_.toString).map(_.toFloat)
      x = Nd4j.concat(0,x,Nd4j.create(Array(concVal)))
    }
    x = Nd4j.hstack(x, Nd4j.ones(total,1))

    (x, y)
  }

  def readFromAFile(infile: String): (INDArray, INDArray) = {

    val numLinesToSkip = 0
    val delimiter = " "
    /**
      *  Reading input and labels from the corresponding data files
      */
    val recordReaderinput = new CSVRecordReader(numLinesToSkip, delimiter)
    val pathsourceinput = new ClassPathResource(infile).getFile
    val file1 = new FileSplit(pathsourceinput)
    recordReaderinput.initialize(file1)

    var concVal = recordReaderinput.next().toArray().map(_.toString).map(_.toFloat)
    var x = Nd4j.create(Array(concVal.slice(1,concVal.length)))
    var y = Nd4j.create(Array(concVal.slice(0,1)))

    /**
      * Storing Data files in  INDArray using n4dj, for running logistic regression
      */
    while (recordReaderinput.hasNext()) {
      concVal = recordReaderinput.next().toArray().map(_.toString).map(_.toFloat)
      x = Nd4j.concat(0,x,Nd4j.create(Array(concVal.slice(1,concVal.length))))
      y = Nd4j.concat(0,y,Nd4j.create(Array(concVal.slice(0,1))))
    }
    (x,y)
  }

}

object main extends App {

  /// READ LABELS AND FILES

  import DataReader._
  /*
  var (x, y) = readFromFiles("src/main/resources/train.txt", "file-1000")
  val iterations = 20000
  var lr = 0.0005f
  val batchsize = 250
  */
  ///* (TOTAL TIME FOR PROCESSING for batchsize: ,250, and iterations ,10000, is ,70.056059099)
  var (x,y) = readFromAFile("cod-rna.txt")
  val iterations = 1200
  var lr = 0.00001f
  val batchsize = 20000
  //*/
  //print(Nd4j.shape(x).toSeq, x.getRow(0), y.sumNumber().doubleValue())


  var ndims = x.columns()

  val model = new SVM(ndims, "zeros")
  model.fit(y, x, lr, iterations, batchsize)
  val y_new = model.classify(x)
  val acc = model.accuracy(y_new, y)
  println(model.loss(y, x), acc * 100)

}
