require "rnn";
require "cunn";
require "cudnn";

function build_model_amar_autoencoders(items_data, ratings_data, genres_data, authors_data, directors_data, properties_data, wiki_categories_data, batch_size)
    local item_embeddings_size = 50
    local genre_embeddings_size = 50
    local author_embeddings_size = 50
    local director_embeddings_size = 50
    local property_embeddings_size = 50
    local wiki_category_embeddings_size = 50
    local half_embeddings_size = item_embeddings_size/2
    local hidden_dense_layer_size = item_embeddings_size -- + user_embeddings_size
    local num_tokens = #items_data["token2id"]
    local lookup_table = nn.LookupTableMaskZero(num_tokens, item_embeddings_size)

    local items_model = nn.Sequential()
            :add(lookup_table)
            :add(nn.SeqBRNN(item_embeddings_size, item_embeddings_size))
            :add(nn.Mean(2))
            :add(nn.Linear(item_embeddings_size, half_embeddings_size))
            :add(nn.Tanh())
            :add(nn.Linear(half_embeddings_size, item_embeddings_size))
            :add(nn.Tanh())
        
    local full_model = nn.Sequential() 

    local parallel_table = nn.ParallelTable()
        :add(items_model)

   if genres_data then
        print("-- Initializing model for genre features")
        hidden_dense_layer_size = hidden_dense_layer_size + genre_embeddings_size
        local genres_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#genres_data["genre2id"] + 1, genre_embeddings_size))
            :add(nn.Mean(2))
            :add(nn.Linear(genre_embeddings_size, half_embeddings_size))
            :add(nn.Tanh())
            :add(nn.Linear(half_embeddings_size, genre_embeddings_size))
            :add(nn.Tanh())

        parallel_table:add(genres_model)
    end

   if authors_data then
        print("-- Initializing model for author features")
        hidden_dense_layer_size = hidden_dense_layer_size + author_embeddings_size
        local authors_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#authors_data["author2id"] + 1, author_embeddings_size))
            :add(nn.Mean(2))
            :add(nn.Linear(author_embeddings_size, half_embeddings_size))
            :add(nn.Tanh())
            :add(nn.Linear(half_embeddings_size, author_embeddings_size))
            :add(nn.Tanh())

        parallel_table:add(authors_model)
    end

   if directors_data then
        print("-- Initializing model for director features")
        hidden_dense_layer_size = hidden_dense_layer_size + director_embeddings_size
        local directors_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#directors_data["director2id"] + 1, director_embeddings_size))
            :add(nn.Mean(2))
            :add(nn.Linear(director_embeddings_size, half_embeddings_size))
            :add(nn.Tanh())
            :add(nn.Linear(half_embeddings_size, director_embeddings_size))
            :add(nn.Tanh())

        parallel_table:add(directors_model)
    end

   if properties_data then
        print("-- Initializing model for property features")
        hidden_dense_layer_size = hidden_dense_layer_size + property_embeddings_size
        local properties_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#properties_data["property2id"] + 1, property_embeddings_size))
            :add(nn.Mean(2))
            :add(nn.Linear(property_embeddings_size, half_embeddings_size))
            :add(nn.Tanh())
            :add(nn.Linear(half_embeddings_size, property_embeddings_size))
            :add(nn.Tanh())

        parallel_table:add(properties_model)
    end

   if wiki_categories_data then
        print("-- Initializing model for wiki category features")
        hidden_dense_layer_size = hidden_dense_layer_size + wiki_category_embeddings_size
        local wiki_categories_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#wiki_categories_data["wiki_category2id"] + 1, wiki_category_embeddings_size))
            :add(nn.Mean(2))
            :add(nn.Linear(wiki_category_embeddings_size, half_embeddings_size))
            :add(nn.Tanh())
            :add(nn.Linear(half_embeddings_size, wiki_category_embeddings_size))
            :add(nn.Tanh())

        parallel_table:add(wiki_categories_model)
    end

    half_embeddings_size = hidden_dense_layer_size/2

    full_model
        :add(parallel_table)
        :add(nn.JoinTable(2))
        :add(nn.Linear(hidden_dense_layer_size, 1))
        :add(cudnn.Sigmoid())

    return full_model
end

