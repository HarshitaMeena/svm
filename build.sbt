name := "svm"

version := "0.1"
scalaVersion := "2.11.8"
resolvers +=
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

val nd4jVersion = "0.8.0"


libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % nd4jVersion

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % nd4jVersion

libraryDependencies += "org.nd4j" %% "nd4s" % nd4jVersion
libraryDependencies += "org.datavec" %  "datavec-api" % nd4jVersion
