require 'nn'
require 'torch'
require 'optim'
require 'image'
require 'paths'
require 'gnuplot'


package.path = package.path ..";".. paths.cwd().."/dataload/?.lua"
--print(package.path)
dl = require 'dataload'


trainLogger = optim.Logger(paths.concat('log', 'train.log'))

function loadData(dataPath)

dataPath = dataPath or "/home/vidh/ILSVRC/Data/DET/"
--print(dataPath)
train, val = dl.loadImageNet(dataPath,1)
print("train")
--[[for key,value in pairs(train) do
   print("found member ".. key.. " ");
   --print (value)
end ]]--
--[[for key, value in pairs(train.dataset) do
--print(train.dataset)
   print( key .."     "..type(value))
end]]--

indices =  torch.LongTensor():range(1, train.dataset.nsample)
inputs, targets = train.dataset:index(indices)



print("val")

indV = torch.LongTensor():range(1, val.dataset.nsample)
inV, tarV = val.dataset:index(indV)
return inputs, targets, inV, tarV, train.dataset.classes
end





trainIn, trainTar, valIn, valTar, classes = loadData()

--print(classes)

-- assume #classes = 1000
dim = trainIn:size()
nSamples = dim[1]
nDim = dim[2]
xPix = dim[3]
yPix = dim[4]

print(dim)

model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3,33,5,5))
model:add(nn.Sigmoid())
model:add(nn.SpatialMaxPooling(3,3,3,3,1,1))

model:add(nn.SpatialConvolutionMM(33,132,3,3))
model:add(nn.Sigmoid())
model:add(nn.SpatialMaxPooling(3,3,3,3,1,1))

model:add(nn.Reshape(24*24*132))
model:add(nn.Linear(24*24*132, 2000))
model:add(nn.Sigmoid())
model:add(nn.Linear(2000, #classes)) 

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
print(model)

--[[ TODO: Normalize data]]--

confusion = optim.ConfusionMatrix(classes)



	params, gradParams = model:getParameters()

function train(inputs, targets)
	epoch = epoch or 1
	print ('training epoch '.. epoch )


	-- [[TODO: add batch processing later]]--    

	local feval = function(x)
		collectgarbage()
		if x~=params then
			params:copy(x)
		end
	  
		gradParams:zero()

		local outputs = model:forward(inputs)
        local f = criterion:forward(outputs, targets)

		local df_do = criterion:backward(outputs, targets)
		model:backward(inputs, df_do)

		outputs = model:forward(inputs)

		for i = 1, inputs:size()[1] do
			confusion:add(outputs[i], targets[i])
		end
		return f, gradParams
	end


	sgdState = sgdState or {
		learningRate = 0.01,
		momentum = 0.005,
		learningRateDecay = 5e-7
	}


	optim.sgd(feval, params, sgdState)
	--xlua.progress()

	print ("trained")
	print(confusion)
	trainLogger:add{['trainAcc'] = confusion.totalValid*100}
	confusion:zero()
	epoch = epoch+1
end


trainIn = trainIn:double()

while true do
	train(trainIn, trainTar)

--	trainLogger:style{['trainAcc'] = '-+'}
--	trainLogger:plot()
end