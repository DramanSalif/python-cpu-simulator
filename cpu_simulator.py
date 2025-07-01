import sys
import logging
import matplotlib.pyplot as plt

# Reconfigure logging to use UTF-8 for both console and file handlers.
# Note: We reopen sys.stdout with UTF-8 encoding.
console_handler = logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler("cpu_simulator.log", mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])


class InstructionMemory:
    def __init__(self, filepath):
        self.instructions = []
        self.load(filepath)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            self.instructions = [line.strip() for line in file.readlines() if line.strip()]

    def get_instruction(self, pc):
        index = pc // 4
        return self.instructions[index] if index < len(self.instructions) else None


class Memory:
    def __init__(self):
        # Simulate memory using a dictionary: key = address, value = data.
        self.data = {}

    def load_word(self, address):
        if address % 4 != 0:
            raise ValueError("Address must be aligned to 4 bytes")
        return self.data.get(address, 0)

    def store_word(self, address, value):
        if address % 4 != 0:
            raise ValueError("Address must be aligned to 4 bytes")
        self.data[address] = value


class Cache:
    def __init__(self):
        self.enabled = False
        self.flushed = False

    def set_cache(self, enabled: bool):
        self.enabled = enabled
        logging.info(f"Cache {'enabled' if self.enabled else 'disabled'}")

    def flush(self):
        self.flushed = True
        logging.info("Cache flushed")


class CPU:
    def __init__(self, instruction_mem):
        self.pc = 0
        self.pc_modified = False  # Flag to indicate if PC was directly modified.
        self.registers = {f"R{i}": 0 for i in range(32)}
        self.imem = instruction_mem
        self.memory = Memory()  # Our simple RAM.
        self.cache = Cache()
        self.running = True
        self.labels = {}

        self.dispatch_table = {
            "ADD": self._handle_add,
            "ADDI": self._handle_addi,
            "SUB": self._handle_sub,
            "SLT": self._handle_slt,
            "BNE": self._handle_bne,
            "J": self._handle_j,
            "JAL": self._handle_jal,
            "LW": self._handle_lw,
            "SW": self._handle_sw,
            "CACHE": self._handle_cache,
            "HALT": self._handle_halt,
        }

    def execute(self):
        while self.running:
            instr = self.imem.get_instruction(self.pc)
            if instr is None:
                break
            self._execute_instruction(instr)
            if self.running:
                if not self.pc_modified:
                    self.pc += 4
                self.pc_modified = False

    def _execute_instruction(self, instr):
        try:
            tokens = instr.replace(';', '').split(',')
            op = tokens[0].strip().upper()
            handler = self.dispatch_table.get(op)
            if handler:
                handler(tokens)
            else:
                raise ValueError(f"Unknown instruction: {op}")
        except Exception as e:
            logging.error(f"Error: {e} in '{instr}'")
            self.running = False

    def _handle_add(self, tokens):
        # Format: ADD,Rd,Rs,Rt
        _, rd, rs, rt = tokens
        self.registers[rd] = self.registers[rs] + self.registers[rt]
        logging.info(f"{rd} = {self.registers[rd]}")

    def _handle_addi(self, tokens):
        # Format: ADDI,Rd,Rs,Imm
        _, rd, rs, imm = tokens
        self.registers[rd] = self.registers[rs] + int(imm)
        logging.info(f"{rd} = {self.registers[rd]}")

    def _handle_sub(self, tokens):
        # Format: SUB,Rd,Rs,Rt
        _, rd, rs, rt = tokens
        self.registers[rd] = self.registers[rs] - self.registers[rt]
        logging.info(f"{rd} = {self.registers[rd]}")

    def _handle_slt(self, tokens):
        # Format: SLT,Rd,Rs,Rt
        _, rd, rs, rt = tokens
        self.registers[rd] = 1 if self.registers[rs] < self.registers[rt] else 0
        logging.info(f"{rd} = {self.registers[rd]}")

    def _handle_bne(self, tokens):
        # Format: BNE,Rs,Rt,Offset
        _, rs, rt, offset = tokens
        if self.registers[rs] != self.registers[rt]:
            self.pc += int(offset) * 4
            self.pc_modified = True
            logging.info(f"Branching to {self.pc}")
        else:
            logging.info("BNE: No branch taken")

    def _handle_j(self, tokens):
        # Format: J,Address
        target = tokens[1].strip()
        self.pc = int(target)
        self.pc_modified = True
        logging.info(f"Jumping to address {self.pc}...")

    def _handle_jal(self, tokens):
        # Format: JAL,TargetLabel or JAL,Address
        target = tokens[1].strip()
        self.registers["R7"] = self.pc + 4
        if target in self.labels:
            self.pc = self.labels[target]
        else:
            self.pc = int(target)
        self.pc_modified = True
        logging.info(f"Jumping to {self.pc} and linking return address in R7 = {self.registers['R7']}")

    def _handle_lw(self, tokens):
        # Format: LW,Rd,Address
        _, rd, address = tokens
        addr = int(address.strip())
        val = self.memory.load_word(addr)
        self.registers[rd] = val
        logging.info(f"{rd} loaded with value {val} from address {addr}")

    def _handle_sw(self, tokens):
        # Format: SW,Rs,Address
        _, rs, address = tokens
        addr = int(address.strip())
        val = self.registers[rs]
        self.memory.store_word(addr, val)
        logging.info(f"Stored {val} from {rs} to memory address {addr}")

    def _handle_cache(self, tokens):
        # Format: CACHE,Value -> "1" enables, "2" flushes, otherwise disable.
        op_val = tokens[1].strip()
        if op_val == "1":
            self.cache.set_cache(True)
        elif op_val == "2":
            self.cache.flush()
        else:
            self.cache.set_cache(False)

    def _handle_halt(self, tokens):
        logging.info("Halting execution.")
        self.running = False

    def preprocess_labels(self):
        self.labels = {}
        cleaned_instructions = []
        for idx, instr in enumerate(self.imem.instructions):
            if ':' in instr:
                label, code = instr.split(':', 1)
                self.labels[label.strip()] = idx * 4
                if code.strip():
                    cleaned_instructions.append(code.strip())
            else:
                cleaned_instructions.append(instr)
        self.imem.instructions = cleaned_instructions

    def display_graphics(self, save_path=None):
        """
        Display a bar chart for registers and memory contents.
        If save_path is provided, the figure will be saved to that location.
        """
        # Plot registers
        regs_names = list(self.registers.keys())
        regs_values = [self.registers[r] for r in regs_names]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(regs_names, regs_values, color='skyblue')
        plt.title("Register Values")
        plt.xlabel("Register")
        plt.ylabel("Value")
        plt.xticks(rotation=90)

        # Plot memory: show only non-zero addresses.
        mem_addresses = sorted(addr for addr, value in self.memory.data.items() if value != 0)
        mem_values = [self.memory.data[addr] for addr in mem_addresses]
        plt.subplot(1, 2, 2)
        if mem_addresses:
            plt.bar([str(addr) for addr in mem_addresses], mem_values, color='lightgreen')
            plt.title("Memory Contents (Non-Zero Values)")
            plt.xlabel("Address")
            plt.ylabel("Value")
        else:
            plt.text(0.5, 0.5, 'Memory is empty', horizontalalignment='center', verticalalignment='center')
            plt.title("Memory Contents")
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Graph saved as {save_path}")

        plt.show()


def run_test():
    logging.info("Running enhanced test suite...")

    # Test 1: Basic arithmetic: ADDI + ADD → R3 should be 15.
    with open("instruction_input.txt", "w", encoding="utf-8") as f:
        f.write("ADDI,R1,R0,5\n")
        f.write("ADDI,R2,R0,10\n")
        f.write("ADD,R3,R1,R2\n")
        f.write("HALT;\n")

    imem = InstructionMemory("instruction_input.txt")
    cpu = CPU(imem)
    cpu.preprocess_labels()
    cpu.execute()
    assert cpu.registers["R3"] == 15, "Test 1 failed: R3 should be 15"
    logging.info("Test 1 passed: Basic arithmetic")

    # Test 2: Label jump and JAL store.
    with open("instruction_input.txt", "w", encoding="utf-8") as f:
        f.write("JAL,target\n")
        f.write("ADD,R4,R0,R0\n")  # Should be skipped.
        f.write("HALT;\n")
        f.write("target: ADDI,R1,R0,9\n")
        f.write("HALT;\n")

    imem = InstructionMemory("instruction_input.txt")
    cpu = CPU(imem)
    cpu.preprocess_labels()
    cpu.execute()
    assert cpu.registers["R1"] == 9, "Test 2 failed: Jumped instruction didn't execute"
    assert cpu.registers["R7"] == 4, "Test 2 failed: R7 should hold return address"
    logging.info("Test 2 passed: JAL with label and link")

    # Test 3: BNE branching.
    with open("instruction_input.txt", "w", encoding="utf-8") as f:
        f.write("ADDI,R1,R0,3\n")
        f.write("ADDI,R2,R0,4\n")
        f.write("BNE,R1,R2,1\n")  # Branch taken since 3 != 4.
        f.write("ADD,R3,R0,R0\n")  # Skipped.
        f.write("ADDI,R3,R0,99\n")
        f.write("HALT;\n")

    imem = InstructionMemory("instruction_input.txt")
    cpu = CPU(imem)
    cpu.preprocess_labels()
    cpu.execute()
    assert cpu.registers["R3"] == 99, "Test 3 failed: Branching error"
    logging.info("Test 3 passed: BNE branching")

    # Test 4: Invalid register name should halt CPU.
    with open("instruction_input.txt", "w", encoding="utf-8") as f:
        f.write("ADDI,R99,R0,5\n")
        f.write("HALT;\n")

    imem = InstructionMemory("instruction_input.txt")
    cpu = CPU(imem)
    cpu.preprocess_labels()
    cpu.execute()
    assert cpu.running is False, "Test 4 failed: CPU should halt on error"
    logging.info("Test 4 passed: Invalid register handling")

    # Test 5: LW and SW operations.
    with open("instruction_input.txt", "w", encoding="utf-8") as f:
        f.write("ADDI,R1,R0,100\n")   # R1 = 100.
        f.write("SW,R1,20\n")          # Store R1 to memory address 20.
        f.write("ADDI,R1,R0,0\n")       # Clear R1.
        f.write("LW,R1,20\n")          # Load value from memory address 20 into R1.
        f.write("HALT;\n")

    imem = InstructionMemory("instruction_input.txt")
    cpu = CPU(imem)
    cpu.preprocess_labels()
    cpu.execute()
    mem_val = cpu.memory.load_word(20)
    assert cpu.registers["R1"] == 100, "Test 5 failed: R1 should be 100 after LW"
    assert mem_val == 100, "Test 5 failed: Memory at address 20 should be 100"
    logging.info("Test 5 passed: LW/SW operations")

    # Test 6: Cache flush behavior.
    with open("instruction_input.txt", "w", encoding="utf-8") as f:
        f.write("CACHE,1\n")  # Enable cache.
        f.write("CACHE,2\n")  # Flush cache.
        f.write("HALT;\n")

    imem = InstructionMemory("instruction_input.txt")
    cpu = CPU(imem)
    cpu.preprocess_labels()
    cpu.execute()
    assert cpu.cache.flushed, "Test 6 failed: Cache should have been flushed"
    logging.info("Test 6 passed: Cache flush behavior")

    logging.info("All tests completed successfully!")


def run_graphical_demo():
    logging.info("Running graphical demo...")
    with open("instruction_input.txt", "w", encoding="utf-8") as f:
        f.write("ADDI,R1,R0,50\n")
        f.write("ADDI,R2,R0,25\n")
        f.write("ADD,R3,R1,R2\n")
        f.write("SW,R3,40\n")
        f.write("LW,R4,40\n")
        f.write("HALT;\n")

    imem = InstructionMemory("instruction_input.txt")
    cpu = CPU(imem)
    cpu.preprocess_labels()
    cpu.execute()
    # Display the final state of registers and memory graphically.
    cpu.display_graphics()


if __name__ == "__main__":
    run_test()
    # Uncomment the next line to launch the graphical demo after testing.
    run_graphical_demo()
