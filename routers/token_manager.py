import os
import asyncio
import time
from typing import List, Dict, Optional
from gradio_client import Client, handle_file
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class TokenManager:
    def __init__(self):
        # Get all possible HF tokens from environment
        self.tokens = []
        for i in range(1, 10):  # Check up to 9 tokens
            token = os.getenv(f"HF{i}")
            if token:
                self.tokens.append(token)
        
        if not self.tokens:
            raise ValueError("No valid tokens found in environment variables")
            
        self.clients = {}
        self.token_status = {}  # True = available, False = busy
        self.token_errors = {}  # Track errors per token
        self.token_quota_exceeded = {}  # Track quota issues
        self.current_operations = {}  # Track current operation per token
        self.request_queue = asyncio.Queue()  # Queue for incoming requests
        self.lock = asyncio.Lock()
        self._last_used_token_index = -1  # For round-robin token selection
        self._processor_tasks: List[asyncio.Task] = []  # multiple workers
        self.initialize_clients()
        
    async def start_processor(self):
        """Start background workers equal to number of tokens"""
        for i in range(len(self.tokens)):
            task = asyncio.create_task(self._request_worker(i))
            self._processor_tasks.append(task)
            logger.info(f"Started worker {i} for token management")
        
    def initialize_clients(self):
        """Initialize clients for all tokens"""
        for i, token in enumerate(self.tokens):
            try:
                client = Client("levihsu/OOTDiffusion", hf_token=token)
                self.clients[token] = client
                self.token_status[token] = True  # Mark as available
                self.token_errors[token] = 0  # Initialize error count
                self.token_quota_exceeded[token] = False  # Initialize quota status
                self.current_operations[token] = None  # No current operation
                logger.info(f"Initialized client for token {i+1}")
            except Exception as e:
                logger.error(f"Failed to initialize client for token {i+1}: {str(e)}")
                self.token_status[token] = False
                self.token_errors[token] = 1
                self.token_quota_exceeded[token] = False
                
    async def _request_worker(self, worker_id: int):
        """Each worker pulls requests and processes them concurrently"""
        while True:
            try:
                request_id, vton_path, garm_path, future = await self.request_queue.get()
                logger.info(f"Worker {worker_id} picked request {request_id}")

                result = await self._process_request(vton_path, garm_path)

                if not future.done():
                    future.set_result(result)

                self.request_queue.task_done()
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_request(self, vton_img_path: str, garm_img_path: str) -> Optional[str]:
        """Process a single request with proper token rotation"""
        # Try each available token once
        for attempt in range(len(self.tokens)):
            token = await self._get_next_available_token()
            if not token:
                logger.error("No available tokens")
                return None
            
            logger.info(f"Attempt {attempt + 1}: Processing with token {token[-10:]}...")
            
            try:
                # Mark token as busy
                async with self.lock:
                    self.token_status[token] = False
                    self.current_operations[token] = "processing"
                
                client = self.clients[token]
                result = await self._call_api_safe(client, vton_img_path, garm_img_path)
                
                if result is not None:
                    # Success!
                    async with self.lock:
                        self.token_status[token] = True
                        self.token_errors[token] = 0
                        self.current_operations[token] = None
                    logger.info(f"Successfully processed with token {token[-10:]}...")
                    return result
                else:
                    # Mark token as quota exceeded and try next token
                    async with self.lock:
                        self.token_quota_exceeded[token] = True
                        self.token_status[token] = False
                        self.current_operations[token] = None
                    logger.warning(f"Token {token[-10:]}... quota exceeded, trying next token")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing with token on attempt {attempt + 1}: {str(e)}")
                
                # Mark token based on error type
                async with self.lock:
                    if "quota" in str(e).lower() or "zerogpu" in str(e):
                        self.token_quota_exceeded[token] = True
                        self.token_status[token] = False
                    else:
                        self.token_errors[token] += 1
                        if self.token_errors[token] >= 3:
                            self.token_status[token] = False
                        else:
                            self.token_status[token] = True
                    self.current_operations[token] = None
                
                # Continue to next token
                continue
        
        logger.error("All tokens failed or have quota exceeded")
        return None
    
    async def _get_next_available_token(self) -> Optional[str]:
        """Get next available token using round-robin selection"""
        start_time = time.time()
        timeout = 10.0  # Wait up to 10 seconds for a token
        
        while time.time() - start_time < timeout:
            async with self.lock:
                # Get all available tokens
                available_tokens = [
                    token for token in self.tokens
                    if (self.token_status.get(token, False) and 
                        not self.token_quota_exceeded.get(token, False) and 
                        self.token_errors.get(token, 0) < 3)
                ]
                
                if available_tokens:
                    # Use round-robin selection
                    self._last_used_token_index = (self._last_used_token_index + 1) % len(available_tokens)
                    selected_token = available_tokens[self._last_used_token_index]
                    
                    # Mark as busy immediately
                    self.token_status[selected_token] = False
                    return selected_token
            
            # If no token available, wait a bit
            await asyncio.sleep(0.1)
        
        return None
    
    async def _call_api_safe(self, client, vton_img_path: str, garm_img_path: str) -> Optional[str]:
        """Make API call with proper error handling"""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None, 
                lambda: client.predict(
                    vton_img=handle_file(vton_img_path),
                    garm_img=handle_file(garm_img_path),
                    n_samples=1,
                    n_steps=20,  
                    image_scale=2,
                    seed=-1,
                    api_name="/process_hd"
                )
            )
            
            return await self._extract_result_path(result)
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            if "quota" in str(e).lower() or "zerogpu" in str(e):
                raise Exception("Quota exceeded") from e
            return None
    
    async def _extract_result_path(self, result) -> Optional[str]:
        """Extract image path from API result"""
        if not result:
            return None
        
        try:
            # The API returns a list of dictionaries
            if isinstance(result, list) and len(result) > 0:
                first_item = result[0]
                if isinstance(first_item, dict) and 'image' in first_item:
                    image_path = first_item['image']
                    if os.path.exists(image_path):
                        return image_path
                    else:
                        logger.warning(f"Image path does not exist: {image_path}")
                        return str(first_item)
                else:
                    return str(first_item)
            elif isinstance(result, str) and os.path.exists(result):
                return result
            elif isinstance(result, dict):
                return str(result)
                
            return str(result)
        except Exception as e:
            logger.error(f"Error extracting result path: {str(e)}")
            return str(result) if result else None
    
    async def process_with_token(self, vton_img_path: str, garm_img_path: str) -> Optional[str]:
        """Public method to process images - adds request to queue"""
        request_id = f"req_{int(time.time() * 1000)}_{id(vton_img_path)}"
        logger.info(f"Adding request {request_id} to queue")
        
        # Create a future for this request
        future = asyncio.Future()
        
        # Add request to queue
        await self.request_queue.put((request_id, vton_img_path, garm_img_path, future))
        
        try:
            # Wait for the result with a timeout
            result = await asyncio.wait_for(future, timeout=120.0)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            if not future.done():
                future.set_exception(TimeoutError("Request timed out"))
            return None
        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
            return None

    def get_usable_tokens_count(self) -> int:
        """Get number of tokens that haven't exceeded quota"""
        return sum(1 for token in self.tokens 
                  if not self.token_quota_exceeded.get(token, False) 
                  and self.token_errors.get(token, 0) < 3
                  and self.token_status.get(token, False))

    def get_service_status(self) -> dict:
        """Get detailed service status"""
        usable_tokens = self.get_usable_tokens_count()
        total_tokens = len(self.tokens)
        
        token_details = []
        for i, token in enumerate(self.tokens):
            token_details.append({
                "token_id": i + 1,
                "status": "available" if self.token_status.get(token, False) else "busy",
                "quota_exceeded": self.token_quota_exceeded.get(token, False),
                "error_count": self.token_errors.get(token, 0),
                "current_operation": self.current_operations.get(token),
                "usable": (not self.token_quota_exceeded.get(token, False) and 
                          self.token_errors.get(token, 0) < 3 and
                          self.token_status.get(token, False))
            })
        
        queue_size = self.request_queue.qsize()
        
        return {
            "total_tokens": total_tokens,
            "usable_tokens": usable_tokens,
            "unusable_tokens": total_tokens - usable_tokens,
            "queue_size": queue_size,
            "service_status": "operational" if usable_tokens > 0 else "quota_exceeded",
            "token_details": token_details
        }

# Global token manager instance
token_manager = TokenManager()
