from typing import Union, Any
from pyfiglet import Figlet
from Core.Chunk.DocChunk import DocChunk
from Core.Common.Logger import logger
import tiktoken
from pydantic import BaseModel, model_validator
from Core.Common.ContextMixin import ContextMixin
from Core.Schema.RetrieverContext import RetrieverContext
from Core.Common.TimeStatistic import TimeStatistic
from Core.Graph import get_graph
from Core.Index import get_index, get_index_config
from Core.Query import get_query
from Core.Storage.NameSpace import Workspace
from Core.Community.ClusterFactory import get_community
from Core.Storage.PickleBlobStorage import PickleBlobStorage
from colorama import Fore, Style, init


init(autoreset=True)  # Initialize colorama and reset color after each print


class GraphRAG(ContextMixin, BaseModel):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    # model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, config):
        super().__init__(config=config)
   
        

    @model_validator(mode="before")
    def welcome_message(cls, values):
        f = Figlet(font='big')  #
        # Generate the large ASCII art text
        logo = f.renderText('DIGIMON')
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        # Print the logo with color
        print(f"{Fore.MAGENTA}{logo}{Style.RESET_ALL}")
        text = [
            "Welcome to DIGIMON: Deep Analysis of Graph-Based RAG Systems.",
            "",
            "Unlock advanced insights with our comprehensive tool for evaluating and optimizing RAG models.",
            "",
            "You can freely combine any graph-based RAG algorithms you desire. We hope this will be helpful to you!"
        ]

        # Function to print a boxed message
        def print_box(text_lines, border_color=Fore.BLUE, text_color=Fore.CYAN):
            max_length = max(len(line) for line in text_lines)
            border = f"{border_color}╔{'═' * (max_length + 2)}╗{Style.RESET_ALL}"
            print(border)
            for line in text_lines:
                print(
                    f"{border_color}║{Style.RESET_ALL} {text_color}{line.ljust(max_length)} {border_color}║{Style.RESET_ALL}")
            border = f"{border_color}╚{'═' * (max_length + 2)}╝{Style.RESET_ALL}"
            print(border)

        # Print the boxed welcome message
        print_box(text)

        # Add a decorative line for separation
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        return values

    @model_validator(mode="after")
    def _update_context(cls, data):
        # cls.config = data.config
        cls.ENCODER = tiktoken.encoding_for_model(data.config.token_model)
        # 不再需要 index_name 作为子目录，因为 working_dir 已经包含方法名
        cls.workspace = Workspace(data.config.working_dir, exp_name=None)  # register workspace
        cls.time_manager = TimeStatistic()
        cls.graph = get_graph(data.config, llm=data.llm, encoder=cls.ENCODER, time_manager=cls.time_manager)  # register graph
        cls.doc_chunk = DocChunk(data.config.chunk, cls.ENCODER, data.workspace.make_for("chunk_storage"), cls.time_manager)
        cls.retriever_context = RetrieverContext()
        data = cls._init_storage_namespace(data)
        data = cls._register_vdbs(data)
        data = cls._register_community(data)
        data = cls._register_e2r_r2c_matrix(data)
        data = cls._register_retriever_context(data)
        return data

    @classmethod
    def _init_storage_namespace(cls, data):
        data.graph.namespace = data.workspace.make_for("graph_storage")
        if data.config.use_entities_vdb:
            data.entities_vdb_namespace = data.workspace.make_for("entities_vdb")
        if data.config.use_relations_vdb:
            data.relations_vdb_namespace = data.workspace.make_for("relations_vdb")
        if data.config.use_subgraphs_vdb:
            data.subgraphs_vdb_namespace = data.workspace.make_for("subgraphs_vdb")
        if data.config.graph.use_community:
            data.community_namespace = data.workspace.make_for("community_storage")
        if data.config.use_entity_link_chunk:
            data.e2r_namespace = data.workspace.make_for("map_e2r")
            data.r2c_namespace = data.workspace.make_for("map_r2c")

   
        return data

    @classmethod
    def _register_vdbs(cls, data):
        # If vector database is needed, register them into the class
        if data.config.use_entities_vdb:
            cls.entities_vdb = get_index(
                get_index_config(data.config, persist_path=data.entities_vdb_namespace.get_save_path()))
        if data.config.use_relations_vdb:
            cls.relations_vdb = get_index(
                get_index_config(data.config, persist_path=data.relations_vdb_namespace.get_save_path()))
        if data.config.use_subgraphs_vdb:
            cls.subgraphs_vdb = get_index(
                get_index_config(data.config, persist_path=data.subgraphs_vdb_namespace.get_save_path()))

        return data

    @classmethod
    def _register_community(cls, data):
        if data.config.graph.use_community:
            cls.community = get_community(data.config.graph.graph_cluster_algorithm,
                                          enforce_sub_communities=data.config.graph.enforce_sub_communities, llm=data.llm,namespace = data.community_namespace
                                         )

        return data

    @classmethod
    def _register_e2r_r2c_matrix(cls, data):
        # The following two matrices are utilized for mapping entities to their corresponding chunks through the specified link-path:
        # Entity Matrix: Represents the entities in the dataset.
        # Chunk Matrix: Represents the chunks associated with the entities.
        # These matrices facilitate the entity -> relationship -> chunk linkage, which is integral to the HippoRAG and FastGraphRAG models.
        if  data.config.graph.graph_type == "tree_graph":
            logger.warning("Tree graph is not supported for entity-link-chunk mapping. Skipping entity-link-chunk mapping.")
            data.config.use_entity_link_chunk = False # Disable entity-link-chunk mapping if tree graph is used.
            return data
        if data.config.use_entity_link_chunk:
            cls.entities_to_relationships = PickleBlobStorage(
                namespace=data.e2r_namespace, config=None
            )
            cls.relationships_to_chunks = PickleBlobStorage(
                namespace=data.r2c_namespace, config=None
            )
        return data

    @classmethod
    def _register_retriever_context(cls, data):
        """
        Register the retriever context based on the configuration provided in `data`.

        Args:
            data: An object containing the configuration.

        Returns:
            The input `data` object.
        """
        cls._retriever_context = {
            "config": True,
            "graph": True,
            "doc_chunk": True,
            "llm": True,
            "entities_vdb": data.config.use_entities_vdb,
            "relations_vdb": data.config.use_relations_vdb,
            "subgraphs_vdb": data.config.use_subgraphs_vdb,
            "community": data.config.graph.use_community,
            "relationships_to_chunks": data.config.use_entity_link_chunk,
            "entities_to_relationships": data.config.use_entity_link_chunk,
        }
        return data

    async def _build_retriever_context(self):
        """
        Build the retriever context for subsequent retriever calls.

        This method registers the necessary contexts with the retriever based on the
        configuration set in `_retriever_context`.
        """
   
        logger.info("Building retriever context for the current execution")
        try:
            for context_name, use_context in self._retriever_context.items():
                if use_context:
                    config_value = getattr(self, context_name)
                    if context_name == "config":
                        config_value = self.config.retriever
                    self.retriever_context.register_context(context_name, config_value)
            self._querier = get_query(self.config.retriever.query_type, self.config.query, self.retriever_context, self.time_manager)

        except Exception as e:
            logger.error(f"Failed to build retriever context: {e}")
            raise


    async def build_e2r_r2c_maps(self, force = False):
        # await self._build_ppr_context()
        logger.info("Starting build two maps: 1️⃣ entity <-> relationship; 2️⃣ relationship <-> chunks ")
        if not await self.entities_to_relationships.load(force):
            await self.entities_to_relationships.set(await self.graph.get_entities_to_relationships_map(False))
            await self.entities_to_relationships.persist()
        if not await self.relationships_to_chunks.load(force):
            await self.relationships_to_chunks.set(await self.graph.get_relationships_to_chunks_map(self.doc_chunk))
            await self.relationships_to_chunks.persist()
        logger.info("✅ Finished building the two maps ")


    def _update_costs_info(self, stage_str:str):
        last_cost = self.llm.get_last_stage_cost()
        logger.info(f"{stage_str} stage cost: Total prompt token: {last_cost.total_prompt_tokens}, Total completeion token: {last_cost.total_completion_tokens}, Total cost: {last_cost.total_cost}")
        last_stage_time = self.time_manager.stop_last_stage()
        logger.info(f"{stage_str} time(s): {last_stage_time:.2f}")

        
    async def insert(self, docs: Union[str, list[Any]]):

        """
        The main function that orchestrates the first step in the Graph RAG pipeline.
        This function is responsible for executing the various stages of the Graph RAG process,
        including chunking, graph construction, index building, and graph augmentation (optional).

        Configuration of the Graph RAG method is based on the parameters provided in the config file.
        For detailed information on the configuration and usage, please refer to the README.md.

        Args:
            docs (Union[str, list[[Any]]): A list of documents to be processed and inserted into the Graph RAG pipeline.
        """


        # Step 1.  Chunking Stage
        self.time_manager.start_stage("chunking")
        self.time_manager.start_named_phase("doc_processing")
        await self.doc_chunk.build_chunks(docs)
        self.time_manager.end_named_phase("doc_processing")
        self._update_costs_info("Chunking")
        self.time_manager.save_stage_details("chunking")


        # Step 2. Building Graph Stage
        self.time_manager.start_stage("build_graph")
        self.time_manager.start_named_phase("graph_construction")
        await self.graph.build_graph(await self.doc_chunk.get_chunks(), self.config.graph.force)
        self.time_manager.end_named_phase("graph_construction")
        self._update_costs_info("Build Graph")
        self.time_manager.save_stage_details("build_graph")

        # Index building Stage (Data-driven content should be pre-built offline to ensure efficient online query performance.)
        self.time_manager.start_stage("index_building")

        # NOTE: ** Ensure the graph is successfully loaded before proceeding to load the index from storage, as it represents a one-to-one mapping. **
        if self.config.use_entities_vdb:
            self.time_manager.start_named_phase("entities_vdb_building")
            node_metadata = await self.graph.node_metadata()
            if not node_metadata:
                logger.warning("No node metadata found. Skipping entity indexing.")

            await self.entities_vdb.build_index(await self.graph.nodes_data(),node_metadata, False)
            self.time_manager.end_named_phase("entities_vdb_building")

        # Graph Augmentation Stage  (Optional)
        # For HippoRAG and MedicalRAG, similarities between entities are utilized to create additional edges.
        # These edges represent similarity types and are leveraged in subsequent processes.

        if self.config.enable_graph_augmentation:
            self.time_manager.start_named_phase("graph_augmentation")
            await self.graph.augment_graph_by_similarity_search(self.entities_vdb)
            self.time_manager.end_named_phase("graph_augmentation")

        if self.config.use_entity_link_chunk:
            self.time_manager.start_named_phase("e2r_r2c_mapping")
            await self.build_e2r_r2c_maps(True)
            self.time_manager.end_named_phase("e2r_r2c_mapping")

        if self.config.use_relations_vdb:
            self.time_manager.start_named_phase("relations_vdb_building")
            edge_metadata = await self.graph.edge_metadata()
            if not edge_metadata:
                logger.warning("No edge metadata found. Skipping relation indexing.")
                return
            await self.relations_vdb.build_index(await self.graph.edges_data(), edge_metadata, force=False)
            self.time_manager.end_named_phase("relations_vdb_building")

        if self.config.use_subgraphs_vdb:
            self.time_manager.start_named_phase("subgraphs_vdb_building")
            subgraph_metadata = await self.graph.subgraph_metadata()
            if not subgraph_metadata:
                logger.warning("No node metadata found. Skipping subgraph indexing.")

            await self.subgraphs_vdb.build_index(await self.graph.subgraphs_data(), subgraph_metadata, force=False)
            self.time_manager.end_named_phase("subgraphs_vdb_building")

        if self.config.graph.use_community:
            self.time_manager.start_named_phase("community_clustering")
            await self.community.cluster(largest_cc=await self.graph.stable_largest_cc(),
                                         max_cluster_size=self.config.graph.max_graph_cluster_size,
                                         random_seed=self.config.graph.graph_cluster_seed, force = False)
            self.time_manager.end_named_phase("community_clustering")

            self.time_manager.start_named_phase("community_report_generation")
            await self.community.generate_community_report(self.graph, False)
            self.time_manager.end_named_phase("community_report_generation")

        self._update_costs_info("Index Building")
        self.time_manager.save_stage_details("index_building")

        # 保存所有阶段的计时数据到文件
        await self._save_timing_metrics()

        await self._build_retriever_context()

   
    
    async def query(self, query):
        """
            Executes the query by extracting the relevant content, and then generating a response.
            Args:
                query: The query to be processed.
            Returns:
        """
        response = await self._querier.query(query)

        # 保存查询阶段的计时数据
        await self._save_timing_metrics()

        return response

    async def _save_timing_metrics(self):
        """保存所有阶段的计时信息到 time_used.json"""
        import json
        import os

        timing_data = self.time_manager.get_stage_details()

        # 如果没有新的计时数据，直接返回
        if not timing_data:
            return

        output_path = os.path.join(self.workspace.get_save_path(), "time_used.json")

        # 追加模式：读取已有数据
        existing_data = []
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing timing data: {e}")
                existing_data = []

        # 追加新数据
        existing_data.extend(timing_data)

        # 写入文件
        try:
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            logger.info(f"Timing metrics saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save timing metrics: {e}")

        # 清空已保存的计时数据，避免重复保存
        self.time_manager.clear_stage_details()
        


   

    
    
   
      
        

   



   

  
  