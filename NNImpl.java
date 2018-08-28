/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes
	
	public ArrayList<Instance> trainingSet=null;//the training set
	
	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. 
 	*/
	
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}
		
		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);
			
		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}
	
	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */
	
	public int calculateOutputForInstance(Instance inst)
	{
		int classIndex = 2;
		double largestValue = -.01;
		ArrayList<Double> outputValues = neuralNetOutput(inst);
		for (int index = 0; index < outputValues.size(); index++)
		{
			double outputValue = roundToOneDecimal(outputValues.get(index));
			if (outputValue >= largestValue)
			{
				largestValue = outputValue;
				classIndex = index;
			}
		}
		return classIndex;
	}
	
	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */
	
	public void train()
	{
		for (int epochNumber = 1; epochNumber <= maxEpoch; epochNumber++)
		{
			for (Instance instance : trainingSet)
			{
				// Update input values in input layer nodes; except bias
				for (int i = 0; i < (inputNodes.size() - 1); i++)
				{
					Node inputNode = inputNodes.get(i);
					double inputValue = instance.attributes.get(i);
					inputNode.setInput(inputValue);
				}
				
				// Forward pass
				ArrayList<Double> outputValues = neuralNetOutput(instance);
				
				// Backward pass
				ArrayList<Integer> teacherValues = instance.classValues;
				double error = 0;
				double deltaHidToOut[] = new double[outputNodes.size()];
				double deltaInToHid[] = new double[hiddenNodes.size()];
				// deltas for weights between hidden and output
				for (int k = 0; k < outputNodes.size(); k++)
				{
					double teacherValue = teacherValues.get(k);
					double outputValue = outputValues.get(k);
					error = teacherValue - outputValue;
					double inK = outputNodes.get(k).getSum();
					deltaHidToOut[k] = error * gPrime(inK);
				}
				// deltas for weights between input and hidden
				for (int j = 0; j < hiddenNodes.size(); j++)
				{
				    double inJ = hiddenNodes.get(j).getSum();
				    deltaInToHid[j] = gPrime(inJ) * sumWJKDeltaK(j, deltaHidToOut);
				}
				
				// Update all w's, starting with weights from hidden to output
				for (int k = 0; k < outputNodes.size(); k++)
				{
					Node outputNode = outputNodes.get(k);
					ArrayList<NodeWeightPair> hiddenLayerNodeWeightPairs = outputNode.parents;
					for (int j = 0; j < hiddenLayerNodeWeightPairs.size(); j++)
					{
						NodeWeightPair hiddenNodeWeightPair = hiddenLayerNodeWeightPairs.get(j);
						Node hiddenNode = hiddenNodeWeightPair.node;
						double weight = hiddenNodeWeightPair.weight;
						double activationJ = hiddenNode.getOutput();
						double newWeight = weight + (learningRate * activationJ * deltaHidToOut[k]);
						hiddenNodeWeightPair.weight = newWeight;
					}
				}
				// Update w's for weights between input and hidden; excluding bias in hidden layer to output layer (since it has no connection to input layer)
				for (int j = 0; j < (hiddenNodes.size() - 1); j++)
				{
					Node hiddenNode = hiddenNodes.get(j);
					ArrayList<NodeWeightPair> inputLayerNodeWeightPairs = hiddenNode.parents;
					for (int i = 0; i < inputLayerNodeWeightPairs.size(); i++)
					{
						NodeWeightPair inputNodeWeightPair = inputLayerNodeWeightPairs.get(i);
						Node inputNode = inputNodeWeightPair.node;
						double weight = inputNodeWeightPair.weight;
						double input = inputNode.getOutput();
						double newWeight = weight + (learningRate * input * deltaInToHid[j]);
						inputNodeWeightPair.weight = newWeight;
					}
				}
			}
		}
	}
	
	private double roundToOneDecimal(double x)
	{
		return ((double) Math.round(x * 10.0)) / 10.0;
	}
	
	/**
	 * Calculates output for instance
	 * 
	 * @param instance  The instance to calculate the output for
	 *  
	 * @return returns the actual output values e.g. [0.1, 0.5, 0.2]
	 */	
	private ArrayList<Double> neuralNetOutput(Instance instance)
	{
		ArrayList<Double> outputValues = new ArrayList<>();
		ArrayList<Double> instanceAttrValues = instance.attributes;
		
		// Read in the input values for the input nodes (excluding the bias) from the instance
		for (int i = 0; i < (inputNodes.size() - 1); i++)
		{
			double x = instanceAttrValues.get(i);
			Node inputNode = inputNodes.get(i);
			inputNode.setInput(x);
		}
		
		// Propagate inputs forward to hidden nodes (excluding bias)
		for (int j = 0; j < (hiddenNodes.size() - 1); j++)
		{
			Node hiddenNode = hiddenNodes.get(j);
			hiddenNode.calculateOutput();
		}
		
		// Propagate inputs forward to output nodes
		for (int k = 0; k < outputNodes.size(); k++)
		{
			Node outputNode = outputNodes.get(k);
			outputNode.calculateOutput();
			outputValues.add(outputNode.getOutput());
		}

		return outputValues;
	}

	/**
	 * Calculate g'(x)
	 * 
	 * @param x  The value to calculate g'(x) for
	 *  
	 * @return g'(x) for the given x 
	 */	
	private double gPrime(double x)
	{
		if (x <= 0)
			return 0;
		else
			return x;
	}

	private double sumWJKDeltaK(int j, double[] deltaHidToOut)
	{
		double sum = 0;
		for (int k = 0; k < outputNodes.size(); k++)
		{
			Node outputNode = outputNodes.get(k);
			double weight = outputNode.parents.get(j).weight;
			sum += (weight * deltaHidToOut[k]);
		}
		return sum;
	}
}
