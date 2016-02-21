local matio = require('matio')


function getDataset(path)
    local data = matio.load(path)
    local graphs, labels, masks, targets = data.conn, data.labels, data.masks, data.targets
    local dataSet = {}

    for i = 1,#graphs do
        local graph, label, mask, target = graphs[i], labels[i], masks[i], targets[i]

        local idx = graph:nonzero()
        local childOfArc, fatherOfArc = idx[{{},1}], idx[{{},2}]

        local nNodes = graph:size(1)
        local nArcs = childOfArc:size(1)

        local childToArcMatrix = torch.zeros(nArcs,nNodes):scatter(2, childOfArc:reshape(nArcs,1), 1)

        mask = mask:nonzero()[{{},1}]
        target = target:index(2, mask)

        dataSet[i] = {
                      {childOfArc, fatherOfArc, childToArcMatrix, nNodes, nArcs, label, mask},
                      target
                     }
    end

    return dataSet
end