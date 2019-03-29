require "datasets/dataset.lua";
require "xlua";
require "cutorch";
require "cunn";
require "cudnn";
require "rnn";
data = require "pl.data";
cjson = require "cjson";
stringx = require "pl.stringx";
require "string";
require "lfs";

title = "### AMAR 2.0 (Ask Me Any Rating) evaluator ###"

print(title)

--[[
We expect to find in the JSON file the following parameters:
    - items: path of items descriptions
    - genres: filename of items genres
    - directors: filename of items genres
    - properties: filename of items genres
    - wiki_categories: filename of items genres
    - models_mapping: dictionary which associates test files to models
    - predictions: generated predictions filename
    - batch_size: number of examples in a batch
    - topn: list of cutoff values
--]]

cmd = torch.CmdLine()
cmd:text()
cmd:text(title)
cmd:text()
cmd:text("Options:")
cmd:option("-config", "", "Filename of JSON training parameters")
cmd:text()

params = cmd:parse(arg)

local conf_data = cjson.decode(file.read(params.config))

print("-- Loading items data: "..conf_data["items"])
items_data = read_items_data(conf_data["items"])

print("-- Padding items data")
items_data["items"] = pad_items_data(items_data)

test_filenames = conf_data["tests"]
model_filenames = conf_data["models"]
batch_size = conf_data["batch_size"]

local genres_data
local authors_data
local directors_data
local properties_data
local wiki_categories_data

if conf_data["genres"] then
    print("-- Loading genres data: ".. conf_data["genres"])
    genres_data = load_items_genres(conf_data["genres"], items_data["item2pos"])

    print("-- Padding genres data")
    genres_data["genres"] = pad_genres_data(genres_data)
end

if conf_data["authors"] then
    print("-- Loading authors data: ".. conf_data["authors"])
    authors_data = load_items_authors(conf_data["authors"], items_data["item2pos"])

    print("-- Padding authors data")
    authors_data["authors"] = pad_authors_data(authors_data)
end

if conf_data["directors"] then
    print("-- Loading directors data: "..conf_data["directors"])
    directors_data = load_items_directors(conf_data["directors"], items_data["item2pos"])

    print("-- Padding directors data")
    directors_data["directors"] = pad_directors_data(directors_data)
end

if conf_data["properties"] then
    print("-- Loading properties data: "..conf_data["properties"])
    properties_data = load_items_properties(conf_data["properties"], items_data["item2pos"])

    print("-- Padding properties data")
    properties_data["properties"] = pad_properties_data(properties_data)
end

if conf_data["wiki_categories"] then
    print("-- Loading wiki_categories data: "..conf_data["wiki_categories"])
    wiki_categories_data = load_items_wiki_categories(conf_data["wiki_categories"], items_data["item2pos"])

    print("-- Padding wiki_categories data")
    wiki_categories_data["wiki_categories"] = pad_wiki_categories_data(wiki_categories_data)
end

for num_folds=1, #test_filenames do
    test_filename = test_filenames[num_folds]
    model_filename = model_filenames[num_folds]

    print("-- Loading test data: "..test_filename)
    local test_data = read_ratings_data(test_filename, items_data["item2pos"], "\t")

    print("-- Loading model: "..model_filename)
    model = torch.load(model_filename)

    local indices = torch.range(1, test_data["ratings"]:size(1)):long():split(batch_size)
    local predictions = {}

    for t, v in ipairs(indices) do
      if t < #indices then
        xlua.progress(t, #indices)
        
        curr_users_batch = test_data["ratings"]:index(1, v)[{ {}, { 1 } }]:cuda()
        curr_items_ids_batch = test_data["ratings"]:index(1, v)[{ {}, { 2 } }]:long()
        local curr_items_ids_batch = torch.reshape(curr_items_ids_batch, v:size(1))
        curr_items_batch = items_data["items"]:index(1, curr_items_ids_batch):cuda()

       if t == #indices then
        last_batch_size = v:size(1)
        curr_users_batch = torch.cat(curr_users_batch, torch.zeros(batch_size-v:size(1), 1):cuda(), 1)
        curr_items_batch = torch.cat(curr_items_batch, torch.zeros(batch_size-v:size(1), items_data["max_item_len"]):cuda(), 1)
       end

        local curr_users_batch = torch.reshape(curr_users_batch, batch_size)

        -- model inputs
        local inputs = { curr_items_batch }
        
        if conf_data["genres"] then
            -- genres ids
            local curr_genres_batch = genres_data["genres"]:index(1, curr_items_ids_batch):cuda()
            table.insert(inputs, curr_genres_batch)
        end

        if conf_data["authors"] then
            -- authors ids
            local curr_authors_batch = authors_data["authors"]:index(1, curr_items_ids_batch):cuda()
            table.insert(inputs, curr_authors_batch)
        end


        if conf_data["directors"] then
            -- directors ids
            local curr_directors_batch = directors_data["directors"]:index(1, curr_items_ids_batch):cuda()
            table.insert(inputs, curr_directors_batch)
        end

        if conf_data["properties"] then
            -- properties ids
            local curr_properties_batch = properties_data["properties"]:index(1, curr_items_ids_batch):cuda()
            table.insert(inputs, curr_properties_batch)
        end

        if conf_data["wiki_categories"] then
            -- wiki_categories ids
            local curr_wiki_categories_batch = wiki_categories_data["wiki_categories"]:index(1, curr_items_ids_batch):cuda()
            table.insert(inputs, curr_wiki_categories_batch)
        end

        local users_batch = curr_users_batch

        local targets = model:forward(inputs)
        
        curr_users_batch = users_batch

        if last_batch_size ~= nil then
          -- remove useless predictions used for batch padding
          targets = targets[{{1, last_batch_size}, {}}]
        end

        for index = 1, targets:size(1) do
            local real_user_id = test_data["id2user"][curr_users_batch[index]]
            if predictions[real_user_id] == nil then
                predictions[real_user_id] = {}
            end
            table.insert(predictions[real_user_id], {
                items_data["pos2item"][curr_items_ids_batch[index]],
                targets[index][1]
            })
        end
     end
    end

    local function cmp_ratings(r1, r2)
        return r1[2] > r2[2]
    end

    for _, topn in pairs(conf_data["topn"]) do
        print("Evaluating predictions for topn: "..topn)
        local results = {}
        
        for user, user_predictions in pairs(predictions) do

            table.sort(user_predictions, cmp_ratings)
            local n = 1
            for _, pair in pairs(user_predictions) do
                local item = pair[1]
                local rating = pair[2]
                table.insert(results, { user, item, rating })
                if n >= topn then
                    break
                end
                n = n + 1
            end
        end
        predictions_filename = string.format(conf_data["predictions"], topn, num_folds)
        
        print("Writing predictions: "..predictions_filename)
        data.write(results, predictions_filename)
    end

    model = nil                                             
    collectgarbage()
end
